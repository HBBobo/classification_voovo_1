import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from typing import List, Dict
import os
import google.generativeai as genai
import asyncio
from dotenv import load_dotenv

from config import constants

load_dotenv()

async def prepare_data_for_training_async(labeled_data: List[Dict]) -> (np.ndarray, np.ndarray, List[str]):
    """
    Prepares the labeled JSON data for training by creating embeddings for the text
    and structuring the scores as a numpy array, asynchronously.
    """
    texts = [item['text'] for item in labeled_data]
    topic_labels = sorted(labeled_data[0]['scores'].keys())
    score_vectors = []
    for item in labeled_data:
        ordered_scores = [item['scores'][label] for label in topic_labels]
        score_vectors.append(ordered_scores)
    y_data = np.array(score_vectors)

    print(f"Generating embeddings for {len(texts)} paragraphs...")
    embedding_result = await genai.embed_content_async(
        model=constants.EMBEDDING_MODEL,
        content=texts,
        task_type=constants.EMBEDDING_TASK_TYPE
    )
    X_data = np.array(embedding_result['embedding'])
    
    return X_data, y_data, topic_labels

def build_and_train_model(X_train, y_train, X_val, y_val, output_labels: List[str]):
    """
    Defines, compiles, and trains the neural network.
    """
    input_shape = (X_train.shape[1],)
    output_shape = y_train.shape[1]

    # --- MÓDOSÍTVA: ÚJ, BÕVÍTETT MODELL ARCHITEKTÚRA ---
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation='relu'), # Extra réteg
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_shape, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.summary()
    
    print("\nStarting model training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )

    print("\nTraining complete. Saving model...")
    os.makedirs(constants.MODELS_DIR, exist_ok=True)
    model.save(constants.STUDENT_MODEL_PATH)
    with open(constants.STUDENT_MODEL_LABELS_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_labels, f)

    return model

async def main():
    """Main async function to run the training process."""
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    try:
        with open(constants.TRAINING_DATASET_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        exit(f"Error: Labeled data not found at '{constants.TRAINING_DATASET_PATH}'. Please run the data generation script first.")

    if not dataset:
        exit("Error: The dataset is empty. Cannot train the model.")

    X_embeddings, y_scores, labels = await prepare_data_for_training_async(dataset)

    X_train, X_val, y_train, y_val = train_test_split(
        X_embeddings, y_scores, test_size=0.2, random_state=42
    )

    build_and_train_model(X_train, y_train, X_val, y_val, labels)
    print(f"\nModel and labels successfully saved to the '{constants.MODELS_DIR}' directory.")

if __name__ == "__main__":
    asyncio.run(main())