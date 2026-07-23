import os
from pathlib import Path

# BASE PATHS
CONFIG_DIR = Path(__file__).parent
BACKEND_DIR = CONFIG_DIR
PROJECT_ROOT = BACKEND_DIR.parent

# Check if running on Render
IS_RENDER = os.environ.get('RENDER', False)

# The HTML files are in Frontend/src/imports/
FRONTEND_DIR = PROJECT_ROOT / "Frontend"  

MODELS_DIR = PROJECT_ROOT / "models"
UPLOAD_FOLDER = BACKEND_DIR / "uploads"
DATASET_DIR = PROJECT_ROOT / "dataset"

# Create directories (if not on Render or if they don't exist)
# On Render, use /tmp for uploads and models
if IS_RENDER:
    UPLOAD_FOLDER = Path('/tmp/uploads')
    MODELS_DIR = Path('/tmp/models')
    DATABASE_URI = 'sqlite:////tmp/analysis.db'
else:
    DATABASE_URI = f'sqlite:///{BACKEND_DIR}/analysis.db'

# FILE UPLOAD
MAX_CONTENT_LENGTH = 100 * 1024 * 1024

ALLOWED_IMAGE_MIMES = {'image/jpeg', 'image/png', 'image/webp', 'image/jpg'}
ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
ALLOWED_VIDEO_MIMES = {'video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/avi', 'video/mpeg', 'video/webm', 'video/x-matroska'}
ALLOWED_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.mpeg'}

# APPLICATION MODES
DEMO_MODE = False

# AUTHENTICATION
JWT_SECRET_KEY = 'your-secret-key-change-this-in-production'
JWT_EXPIRATION_HOURS = 24
FREE_TRIAL_LIMIT = 1

# GEMINI API
GEMINI_API_KEY = ''
GEMINI_MODEL = "gemini-1.5-flash"
ENABLE_EXPLANATIONS = False

# VIDEO SETTINGS
MAX_VIDEO_DURATION_SECONDS = 300
FRAME_SAMPLE_INTERVAL = 10
MAX_FRAMES_TO_ANALYZE = 200

# FEEDBACK & RETRAINING
RETRAIN_THRESHOLD = 50

# TRAINING
IMG_SIZE = 224
BATCH_SIZE = 128
EPOCHS = 50
NUM_FEATURES = 2048
SMALL_DATASET_THRESHOLD = 5000
MEDIUM_DATASET_THRESHOLD = 20000

ENSEMBLE_WEIGHTS = {'resnet50': 0.4, 'efficientnet': 0.4, 'vit': 0.2}

# CLAMAV
CLAMAV_ENABLED = False
CLAMAV_SOCKET = '/var/run/clamav/clamd.ctl'
CLAMAV_HOST = 'localhost'
CLAMAV_PORT = 3310


def print_config():
    print("=" * 60)
    print("CONFIGURATION LOADED")
    print("=" * 60)
    print(f"FRONTEND_DIR: {FRONTEND_DIR}")
    print(f"UPLOAD_FOLDER: {UPLOAD_FOLDER}")
    print(f"FREE_TRIAL_LIMIT: {FREE_TRIAL_LIMIT}")
    print("=" * 60)
