import os
import glob
import json
import numpy as np
import random
import asyncio
from typing import List, Dict, Any, Tuple, Optional
import google.generativeai as genai

# --- Component Imports ---
from utils.universal_parser import extract_paragraphs_from_file_async
from utils.create_training_data import score_paragraph_universally_async as get_scores_from_oracle_async
from utils.create_training_data import load_cache as load_oracle_cache
from utils.create_training_data import save_cache as save_oracle_cache

from dotenv import load_dotenv

# --- Project-wide Constants ---
from config import constants

load_dotenv()


def load_student_nn_model_and_labels() -> Tuple[Optional[Any], Optional[List[str]]]:
    """Loads a trained Keras model and its labels if they exist."""
    try:
        import tensorflow as tf
    except ImportError:
        print("Warning: TensorFlow is not installed. The system will rely solely on the Oracle.")
        return None, None
        
    if not os.path.exists(constants.STUDENT_MODEL_PATH) or not os.path.exists(constants.STUDENT_MODEL_LABELS_PATH):
        print("Warning: Trained student model or labels not found. System will start in Oracle-only mode.")
        return None, None
    
    print(f"-> Loading trained student model from '{constants.STUDENT_MODEL_PATH}'...")
    model = tf.keras.models.load_model(constants.STUDENT_MODEL_PATH)
    with open(constants.STUDENT_MODEL_LABELS_PATH, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    return model, labels

async def retrain_student_model_async() -> Tuple[Optional[Any], Optional[List[str]]]:
    """
    Combines new labels with the full cached dataset and retrains the NN asynchronously.
    """
    try:
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("Error: TensorFlow or Scikit-learn not installed. Cannot retrain model.")
        return None, None
    
    full_dataset_dict = load_oracle_cache(constants.ORACLE_CACHE_PATH)
    if not full_dataset_dict or len(full_dataset_dict) < constants.RETRAIN_TRIGGER_COUNT:
        print(f"Retraining skipped: Not enough samples in Oracle cache ({len(full_dataset_dict)}).")
        return None, None
    
    full_dataset = list(full_dataset_dict.values())
    print(f"\n--- Retraining Triggered! Total training samples: {len(full_dataset)} ---")

    texts = [item['text'] for item in full_dataset]
    topic_labels = sorted(full_dataset[0]['scores'].keys())
    y_data = np.array([[item['scores'].get(label, 0.0) for label in topic_labels] for item in full_dataset])
    
    print("Generating embeddings for the full dataset...")
    embedding_result = await genai.embed_content_async(
        model=constants.EMBEDDING_MODEL, 
        content=texts, 
        task_type=constants.EMBEDDING_TASK_TYPE
    )
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

    os.makedirs(constants.MODELS_DIR, exist_ok=True)
    print(f"Retraining complete. Saving new model to '{constants.STUDENT_MODEL_PATH}'...")
    model.save(constants.STUDENT_MODEL_PATH)
    with open(constants.STUDENT_MODEL_LABELS_PATH, 'w', encoding='utf-8') as f:
        json.dump(topic_labels, f)

    print("--- Resuming main pipeline with the new, smarter model. ---\n")
    return model, topic_labels

async def generate_single_embedding_async(text: str) -> Optional[np.ndarray]:
    """Generates an embedding for a single piece of text asynchronously."""
    try:
        result = await genai.embed_content_async(
            model=constants.EMBEDDING_MODEL, 
            content=text, 
            task_type=constants.EMBEDDING_TASK_TYPE
        )
        return np.array(result['embedding'])
    except Exception as e:
        print(f"    -> Error generating embedding: {e}")
        return None

def predict_with_student_nn(model: Any, paragraph_embedding: np.ndarray, model_labels: List[str]) -> Dict[str, float]:
    """Makes a prediction with our fast, local neural network."""
    scores_vector = model.predict(np.expand_dims(paragraph_embedding, axis=0), verbose=0)[0]
    return {label: float(score) for label, score in zip(model_labels, scores_vector)}

async def run_pipeline_async():
    """
    The main async function that runs the active learning pipeline.
    It processes paragraphs, collects uncertain ones into batches,
    queries the Oracle in parallel for these batches, and retrains the
    student model immediately after.
    """
    print("--- Starting Active Learning Pipeline ---")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    student_model, model_labels = load_student_nn_model_and_labels()
    oracle_cache = load_oracle_cache(constants.ORACLE_CACHE_PATH)
    with open(constants.TOPICS_FILE_PATH, 'r', encoding='utf-8') as f:
        all_topics_json = json.load(f)['topics']

    print(f"\nAggregating all paragraphs from folder: '{constants.INPUT_DOCUMENT_FOLDER}'...")
    all_paragraphs_with_origin = []
    files_to_process = glob.glob(f"{constants.INPUT_DOCUMENT_FOLDER}/*")
    
    parser_tasks = [extract_paragraphs_from_file_async(file_path) for file_path in files_to_process]
    all_paragraph_lists = await asyncio.gather(*parser_tasks)

    for file_path, paragraphs in zip(files_to_process, all_paragraph_lists):
        for p in paragraphs:
            all_paragraphs_with_origin.append({"text": p, "origin_file": os.path.basename(file_path)})
    
    if not all_paragraphs_with_origin:
        exit("No paragraphs found in any documents. Exiting.")

    print(f"Found a total of {len(all_paragraphs_with_origin)} paragraphs. Shuffling...")
    random.shuffle(all_paragraphs_with_origin)

    final_results = []
    # --- ÚJ: Várólista az Orákulum számára ---
    oracle_waitlist = []
    total_oracle_calls = 0

    async def process_oracle_batch(batch_to_process):
        """Helper function to process a batch and retrain the model."""
        nonlocal student_model, model_labels, total_oracle_calls

        if not batch_to_process:
            return

        print(f"\n--- Waitlist full. Calling Oracle in parallel for {len(batch_to_process)} paragraphs ---")
        tasks = [get_scores_from_oracle_async(p_info["text"], all_topics_json) for p_info in batch_to_process]
        oracle_results = await asyncio.gather(*tasks)
        
        total_oracle_calls += len(oracle_results)
        print(f"--- Oracle batch processing complete. ---")

        newly_collected_labels = []
        for para_info, result_package in zip(batch_to_process, oracle_results):
            final_scores = result_package.get("result", {}).get("scores", {})
            if final_scores:
                new_label = {"text": para_info["text"], "scores": final_scores}
                oracle_cache[para_info["text"]] = new_label
                newly_collected_labels.append(new_label)
            
            # A végeredmény listához azonnal hozzáadjuk
            final_results.append({"file": para_info["origin_file"], "text": para_info["text"], "scores": final_scores})

        if newly_collected_labels:
             save_oracle_cache(constants.ORACLE_CACHE_PATH, oracle_cache)
             print("    -> Saved new labels to Oracle cache.")
        
        # --- Azonnali Újratanítás a Batch után ---
        print(f"\n>>> Triggering mid-run retraining with {len(newly_collected_labels)} new labels... <<<")
        new_model, new_labels = await retrain_student_model_async()
        if new_model and new_labels:
            student_model = new_model
            model_labels = new_labels
            print(">>> Retraining complete. Pipeline continues with the smarter model. <<<\n")


    # --- Fõ Feldolgozási Ciklus ---
    for i, para_info in enumerate(all_paragraphs_with_origin):
        para_text = para_info["text"]
        origin = para_info["origin_file"]
        print(f"\nProcessing paragraph {i+1}/{len(all_paragraphs_with_origin)} (from {origin})...")
        
        # 1. Gyorsítótár ellenõrzése
        if para_text in oracle_cache:
            print("  -> [ORACLE CACHE] HIT. Using cached scores.")
            final_scores = oracle_cache[para_text]['scores']
            final_results.append({"file": origin, "text": para_text, "scores": final_scores})
            continue

        # 2. Diák modell próbálkozás
        use_oracle = True
        if student_model and model_labels:
            para_embedding = await generate_single_embedding_async(para_text)
            if para_embedding is not None:
                student_scores = predict_with_student_nn(student_model, para_embedding, model_labels)
                confidence = np.max(list(student_scores.values()))
                
                if confidence >= constants.CONFIDENCE_THRESHOLD:
                    use_oracle = False
                    print(f"  -> [NN] High Confidence ({confidence:.2f}). Using Student's scores.")
                    final_results.append({"file": origin, "text": para_text, "scores": student_scores})
        
        # 3. Ha az Orákulum kell, a várólistára tesszük
        if use_oracle:
            if not student_model: print("  -> [ORACLE] No student model. Adding to waitlist...")
            else: print(f"  -> [ORACLE] Low Confidence. Adding to waitlist...")
            
            oracle_waitlist.append(para_info)
        
        # 4. Ellenõrizzük, hogy a várólista betelt-e
        if len(oracle_waitlist) >= constants.RETRAIN_TRIGGER_COUNT:
            await process_oracle_batch(oracle_waitlist)
            # A batch feldolgozása után kiürítjük a várólistát
            oracle_waitlist = []

    # --- Ciklus utáni maradék feldolgozása ---
    # Ha a ciklus végén maradt valami a várólistán (kevesebb mint 25), azt is feldolgozzuk.
    if oracle_waitlist:
        print("\n--- Processing remaining paragraphs on the waitlist... ---")
        await process_oracle_batch(oracle_waitlist)
        oracle_waitlist = []

    print("\n--- Pipeline Complete ---")
    print(f"Processed {len(files_to_process)} files and {len(final_results)} paragraphs.")
    print(f"The Oracle (Gemini API) was called a total of {total_oracle_calls} times.")

    print(f"Saving final classified results to '{constants.FINAL_RESULTS_PATH}'...")
    os.makedirs(os.path.dirname(constants.FINAL_RESULTS_PATH), exist_ok=True)
    with open(constants.FINAL_RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    print("\nProcess finished.")