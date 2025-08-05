import os
import glob
import json
import numpy as np
import random
import time
from typing import List, Dict, Any, Tuple, Optional

# --- Component Imports ---
# Ensure these scripts are in the same directory or accessible in your PYTHONPATH
from universal_parser import extract_paragraphs_from_file
from create_training_data import score_paragraph_universally as get_scores_from_oracle
from create_training_data import load_cache as load_oracle_cache
from create_training_data import save_cache as save_oracle_cache

from dotenv import load_dotenv
load_dotenv()

INPUT_DOCUMENT_FOLDER = "data/content/"
MODEL_DIR = "models/"
MODEL_PATH = os.path.join(MODEL_DIR, "paragraph_classifier.h5")
MODEL_LABELS_PATH = os.path.join(MODEL_DIR, "model_labels.json")
ORACLE_CACHE_PATH = "data/oracle_scores_cache.json"
FINAL_RESULTS_PATH = "data/final_classified_results.json"

CONFIDENCE_THRESHOLD = 0.85  # If NN's top score is below this, ask the Oracle.
RETRAIN_TRIGGER_COUNT = 25   # Retrain the model every time we collect this many new labels.
EMBEDDING_MODEL = 'models/embedding-001'


def load_student_nn_model_and_labels() -> Tuple[Optional[Any], Optional[List[str]]]:
    """Loads a trained Keras model and its labels if they exist."""
    try:
        import tensorflow as tf
    except ImportError:
        print("Warning: TensorFlow is not installed. The system will rely solely on the Oracle.")
        return None, None
        
    if not os.path.exists(MODEL_PATH) or not os.path.exists(MODEL_LABELS_PATH):
        print("Warning: Trained student model or labels not found. System will start in Oracle-only mode.")
        return None, None
    
    print(f"-> Loading trained student model from '{MODEL_PATH}'...")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(MODEL_LABELS_PATH, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    return model, labels

def retrain_student_model() -> Tuple[Optional[Any], Optional[List[str]]]:
    """
    Combines new labels with the full cached dataset and retrains the NN.
    This function encapsulates the logic of the 'train_student_model.py' script.
    """
    try:
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("Error: TensorFlow or Scikit-learn not installed. Cannot retrain model.")
        return None, None
    
    full_dataset_dict = load_oracle_cache(ORACLE_CACHE_PATH)
    if not full_dataset_dict or len(full_dataset_dict) < RETRAIN_TRIGGER_COUNT:
        print(f"Retraining skipped: Not enough samples in Oracle cache ({len(full_dataset_dict)}).")
        return None, None
    
    full_dataset = list(full_dataset_dict.values())
    print(f"\n--- Retraining Triggered! Total training samples: {len(full_dataset)} ---")

    texts = [item['text'] for item in full_dataset]
    topic_labels = sorted(full_dataset[0]['scores'].keys())
    y_data = np.array([[item['scores'].get(label, 0.0) for label in topic_labels] for item in full_dataset])
    
    print("Generating embeddings for the full dataset...")
    embedding_result = genai.embed_content(model=EMBEDDING_MODEL, content=texts, task_type="CLASSIFICATION")
    X_data = np.array(embedding_result['embedding'])

    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("Starting retraining...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
              verbose=0)

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Retraining complete. Saving new model to '{MODEL_PATH}'...")
    model.save(MODEL_PATH)
    with open(MODEL_LABELS_PATH, 'w', encoding='utf-8') as f:
        json.dump(topic_labels, f)

    print("--- Resuming main pipeline with the new, smarter model. ---\n")
    return model, topic_labels

def generate_single_embedding(text: str) -> Optional[np.ndarray]:
    """Generates an embedding for a single piece of text."""
    try:
        result = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="retrieval_document")
        return np.array(result['embedding'])
    except Exception as e:
        print(f"    -> Error generating embedding: {e}")
        return None

def predict_with_student_nn(model: Any, paragraph_embedding: np.ndarray, model_labels: List[str]) -> Dict[str, float]:
    """Makes a prediction with our fast, local neural network."""
    scores_vector = model.predict(np.expand_dims(paragraph_embedding, axis=0), verbose=0)[0]
    return {label: float(score) for label, score in zip(model_labels, scores_vector)}

if __name__ == "__main__":
    import google.generativeai as genai
    print("--- Starting Active Learning Pipeline ---")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    student_model, model_labels = load_student_nn_model_and_labels()
    oracle_cache = load_oracle_cache(ORACLE_CACHE_PATH)
    with open("data/main_topics_subtopics.json", 'r', encoding='utf-8') as f:
        all_topics_json = json.load(f)['topics']

    print(f"\nAggregating all paragraphs from folder: '{INPUT_DOCUMENT_FOLDER}'...")
    all_paragraphs_with_origin = []
    files_to_process = glob.glob(f"{INPUT_DOCUMENT_FOLDER}/*")
    
    for file_path in files_to_process:
        paragraphs = extract_paragraphs_from_file(file_path)
        for p in paragraphs:
            all_paragraphs_with_origin.append({"text": p, "origin_file": os.path.basename(file_path)})
    
    if not all_paragraphs_with_origin:
        exit("No paragraphs found in any documents. Exiting.")

    print(f"Found a total of {len(all_paragraphs_with_origin)} paragraphs. Shuffling...")
    random.shuffle(all_paragraphs_with_origin)

    final_results = []
    newly_collected_labels_for_retraining = []
    oracle_call_count = 0

    for i, para_info in enumerate(all_paragraphs_with_origin):
        para_text = para_info["text"]
        origin = para_info["origin_file"]
        print(f"\nProcessing paragraph {i+1}/{len(all_paragraphs_with_origin)} (from {origin})...")
        
        use_oracle = True
        final_scores = {}
        
        if student_model and model_labels:
            para_embedding = generate_single_embedding(para_text)
            if para_embedding is not None:
                student_scores = predict_with_student_nn(student_model, para_embedding, model_labels)
                confidence = np.max(list(student_scores.values()))
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    use_oracle = False
                    final_scores = student_scores
                    print(f"  -> [NN] High Confidence ({confidence:.2f}). Using Student's scores.")

        if use_oracle:
            if not student_model: print("  -> [ORACLE] No student model. Escalating...")
            else: print(f"  -> [ORACLE] Low Confidence. Escalating to Gemini...")
            
            oracle_call_count += 1
            
            if para_text in oracle_cache:
                print("    -> Oracle Cache HIT.")
                final_scores = oracle_cache[para_text]['scores']
            else:
                print("    -> Oracle Cache MISS. Calling API...")
                oracle_result = get_scores_from_oracle(para_text, all_topics_json)
                final_scores = oracle_result.get("scores", {})
                
                if final_scores:
                    new_label = {"text": para_text, "scores": final_scores}
                    oracle_cache[para_text] = new_label
                    newly_collected_labels_for_retraining.append(new_label)
                    save_oracle_cache(ORACLE_CACHE_PATH, oracle_cache)
                    print("    -> Saved new label to Oracle cache.")

        final_results.append({"file": origin, "text": para_text, "scores": final_scores})

        if len(newly_collected_labels_for_retraining) >= RETRAIN_TRIGGER_COUNT:
            new_model, new_labels = retrain_student_model()
            if new_model:
                student_model = new_model
                model_labels = new_labels
            newly_collected_labels_for_retraining = []

    print("\n--- Pipeline Complete ---")
    print(f"Processed {len(files_to_process)} files and {len(final_results)} paragraphs.")
    print(f"The Oracle (Gemini API) was called {oracle_call_count} times.")

    print(f"Saving final classified results to '{FINAL_RESULTS_PATH}'...")
    os.makedirs(os.path.dirname(FINAL_RESULTS_PATH), exist_ok=True)
    with open(FINAL_RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    print("\nProcess finished.")