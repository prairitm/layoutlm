# Configuration variables such as file paths, model settings, etc.


# General Project Directory
PROJECT_DIR = '/home/prairit/Documents/layoutlm'

# Data Directories
INPUT_IMAGES_DIR = f'{PROJECT_DIR}/input_images'
OUTPUT_DIR = f'{PROJECT_DIR}/output'
LAYOUTLMV3_HOCR_OUTPUT_DIR = f'{PROJECT_DIR}/layoutlmv3_hocr_output'

# JSON Data
LABEL_STUDIO_JSON_PATH = f'{PROJECT_DIR}/project-1-at-2024-03-15-07-31-acf1c3d2.json'

# Model Configuration
MODEL_NAME = 'microsoft/layoutlmv3-base'
TRAINING_OUTPUT_DIR = f'{PROJECT_DIR}/training_output'

# Training Configuration
TRAIN_TEST_SPLIT_RATIO = 0.3
RANDOM_STATE = 21
MAX_TRAINING_STEPS = 1000
EVALUATION_STRATEGY = 'steps'
EVALUATION_STEPS = 100
LEARNING_RATE = 1e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = 'f1'

# Inference Configuration
INFER_OUTPUT_DIR = f'{PROJECT_DIR}/inference_output'

# Visualization Configuration
LABEL2COLOR = {
    "SNo.": 'blue',
    "Case No.": 'green',
    "Petitioner / Respondent": 'orange',
    "Petitioner/Respondent Advocate": 'red',
    "Case Order": 'purple'
}

# File Paths for Saving/Loading Processed Data
FINAL_LIST_TXT_PATH = f'{OUTPUT_DIR}/final_list_text.txt'
TRAIN_TXT_PATH = f'{OUTPUT_DIR}/train.txt'
TEST_TXT_PATH = f'{OUTPUT_DIR}/test.txt'

# Install Commands (Optional, for documentation or script use)
INSTALL_COMMANDS = [
    'pip install pytesseract',
    'apt-get update',
    'apt install tesseract-ocr',
    'apt install libtesseract-dev'
]

# Add any other global constants or configuration variables as needed
