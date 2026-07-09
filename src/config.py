"""
Configuration file for Infant Cry Classification Project
"""

import os
import torch

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
BALANCED_DATA_DIR = os.path.join(DATA_DIR, "balanced")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
AUGMENTED_DATA_DIR = os.path.join(DATA_DIR, "augmented")
FEATURE_DATA_DIR = os.path.join(DATA_DIR, "features")
SPLIT_DATA_DIR = os.path.join(DATA_DIR, "splits")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_infant_cry_resnet.pth")
# ==========================================================
# DATASET
# ==========================================================

CLASS_NAMES = [
    "hungry",
    "tired",
    "discomfort",
    "belly_pain",
    "burping"
]

CLASS_TO_LABEL = {
    "hungry": 0,
    "tired": 1,
    "discomfort": 2,
    "belly_pain": 3,
    "burping": 4
}

SUPPORTED_AUDIO_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".ogg",
    ".m4a"
)

# ==========================================================
# BALANCING
# ==========================================================

RANDOM_SEED = 42

OVERSAMPLE = True

DUPLICATE_SUFFIX = "_copy"

REMOVE_EXISTING_BALANCED_DATA = True

# Dataset
# ================================================================

NUM_CLASSES = 5

AUDIO_DURATION = 5

N_FFT = 1024

HOP_LENGTH = 512

N_MELS = 64

WEIGHT_DECAY = 1e-4

PIN_MEMORY = True

NUM_WORKERS = 2

MODEL_NAME = "resnet34"



# ==========================================================
# AUDIO
# ==========================================================

TARGET_SAMPLE_RATE = 16000

AUDIO_DURATION = 5

N_MELS = 64

N_FFT = 1024

HOP_LENGTH = 512



# ==========================================================
# MODEL
# ==========================================================

MODEL_NAME = "resnet34"

NUM_CLASSES = 5

DROPOUT = 0.5

PRETRAINED = True

BEST_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "best_infant_cry_resnet.pth"
)


# ==========================================================
# TRAINING
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32

EPOCHS = 75

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

TRAIN_SPLIT = 0.80

VAL_SPLIT = 0.20

NUM_WORKERS = 2

PIN_MEMORY = True

RANDOM_SEED = 42

LR_FACTOR = 0.5

LR_PATIENCE = 4

SAVE_BEST_ONLY = True

# ==========================================================
# INFERENCE
# ==========================================================

CLASS_LABELS = {
    0: "Hungry",
    1: "Tired",
    2: "Discomfort",
    3: "Belly Pain",
    4: "Burping"
}

# ==========================================================
# EVALUATION
# ==========================================================

CONFUSION_MATRIX_PATH = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

CLASSIFICATION_REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "metrics.json"
)

PREDICTIONS_PATH = os.path.join(
    RESULTS_DIR,
    "predictions.csv"
)