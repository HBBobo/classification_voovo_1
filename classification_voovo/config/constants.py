import os

# --- DIRECTORIES AND FILE PATHS ---

# Base directories
DATA_DIR = "data"
MODELS_DIR = "models"
INPUT_DOCUMENT_FOLDER = os.path.join(DATA_DIR, "content+")

# Data and result files
TOPICS_FILE_PATH = os.path.join(DATA_DIR, "main_topics_subtopics.json")
ORACLE_CACHE_PATH = os.path.join(DATA_DIR, "oracle_scores_cache.json")
FINAL_RESULTS_PATH = os.path.join(DATA_DIR, "final_classified_results.json")
TRAINING_DATASET_PATH = os.path.join(DATA_DIR, "training_dataset_universal_scores.json")

# Model files
STUDENT_MODEL_FILENAME = "paragraph_classifier.h5"
STUDENT_MODEL_LABELS_FILENAME = "model_labels.json"
STUDENT_MODEL_PATH = os.path.join(MODELS_DIR, STUDENT_MODEL_FILENAME)
STUDENT_MODEL_LABELS_PATH = os.path.join(MODELS_DIR, STUDENT_MODEL_LABELS_FILENAME)


# --- MODEL AND PIPELINE PARAMETERS ---

# Gemini model identifiers
ORACLE_MODEL = 'gemini-2.5-pro'
PARAGRAPH_SPLITTER_MODEL = 'gemini-2.5-flash-lite'
EMBEDDING_MODEL = 'gemini-embedding-001'


# Active learning pipeline thresholds
CONFIDENCE_THRESHOLD = 0.8   # If NN's top score is below this, ask the Oracle.
RETRAIN_TRIGGER_COUNT = 25   # Retrain the model every time we collect this many new labels.


# --- EMBEDDING SETTINGS ---

# The task type provided to the embedding model. This can influence the quality of the vector.
# Possible values: "CLASSIFICATION", "RETRIEVAL_DOCUMENT", "CLUSTERING", "SEMANTIC_SIMILIRITY"

# Task type used for training and retraining the model.
EMBEDDING_TASK_TYPE = "CLASSIFICATION"