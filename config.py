import os
from pathlib import Path

# Base project directory (root of the repository)
BASE_DIR = Path(__file__).resolve().parent

# Data paths
RAW_DATA_SETA = str(BASE_DIR / 'data' / 'raw' / 'training_setA')
RAW_DATA_SETB = str(BASE_DIR / 'data' / 'raw' / 'training_setB')
PROCESSED_DATA = str(BASE_DIR / 'data' / 'processed')

# Model paths
MODELS_DIR = str(BASE_DIR / 'models')

# Results paths
RESULTS_FIGURES = str(BASE_DIR / 'results' / 'figures')
RESULTS_METRICS = str(BASE_DIR / 'results' / 'metrics')
RESULTS_LOGS = str(BASE_DIR / 'results' / 'logs')

# Subdirectories for figures
FIGURES_EXPLORATION = str(Path(RESULTS_FIGURES) / 'data_exploration')
FIGURES_TRAINING = str(Path(RESULTS_FIGURES) / 'training')
FIGURES_EVALUATION = str(Path(RESULTS_FIGURES) / 'evaluation')

# Model parameters
WINDOW_SIZE = 12
STEP_SIZE = 1
RANDOM_SEED = 42
# Base project directory
BASE_DIR = '/Users/robertchitic/Desktop/AI Coursework'

# Data paths
RAW_DATA_SETA = os.path.join(BASE_DIR, 'data', 'raw', 'training_setA')
RAW_DATA_SETB = os.path.join(BASE_DIR, 'data', 'raw', 'training_setB')
PROCESSED_DATA = os.path.join(BASE_DIR, 'data', 'processed')

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Results paths
RESULTS_FIGURES = os.path.join(BASE_DIR, 'results', 'figures')
RESULTS_METRICS = os.path.join(BASE_DIR, 'results', 'metrics')
RESULTS_LOGS = os.path.join(BASE_DIR, 'results', 'logs')

# Subdirectories for figures
FIGURES_EXPLORATION = os.path.join(RESULTS_FIGURES, 'data_exploration')
FIGURES_TRAINING = os.path.join(RESULTS_FIGURES, 'training')
FIGURES_EVALUATION = os.path.join(RESULTS_FIGURES, 'evaluation')

# Model parameters
WINDOW_SIZE = 12
STEP_SIZE = 1
RANDOM_SEED = 42