import os
from pathlib import Path

# Base paths (relative to this file)
BASE_DIR = Path(__file__).parent.parent
BACKEND_DIR = Path(__file__).parent
MODELS_DIR = BACKEND_DIR.parent / "models"
UPLOAD_FOLDER = BACKEND_DIR / "uploads"
DATASET_DIR = BACKEND_DIR.parent / "dataset"

# Database
DATABASE_URI = os.environ.get('DATABASE_URL', f'sqlite:///{BACKEND_DIR}/analysis.db')

# File limits
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB
ALLOWED_IMAGE_MIMES = {'image/jpeg', 'image/png'}
ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}

# Demo mode (random predictions)
DEMO_MODE = os.environ.get('DEMO_MODE', 'false').lower() == 'true'

# ClamAV (set CLAMAV_ENABLED=true to enable virus scanning)
CLAMAV_ENABLED = os.environ.get('CLAMAV_ENABLED', 'false').lower() == 'true'
CLAMAV_SOCKET = '/var/run/clamav/clamd.ctl'

# Retraining threshold
RETRAIN_THRESHOLD = 50

# Create directories
UPLOAD_FOLDER.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)