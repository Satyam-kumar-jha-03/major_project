#!/usr/bin/env python
"""
Retraining Pipeline for AI vs Human Detector - COMPLETE VERSION
Triggered when feedback threshold is reached.
Uses corrected predictions to fine-tune the model.
"""
import sys
import json
import time
import gc
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras

sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS_DIR, BACKEND_DIR, RETRAIN_THRESHOLD


def collect_corrected_dataset():
    """Fetch all feedback with corrected_label and create a CSV for retraining."""
    print("Collecting corrected feedback data...")

    try:
        from app import app
        from models import db, Feedback, MediaAnalysis
    except ImportError:
        print("Cannot import app modules. Make sure you're in the backend directory.")
        return False

    with app.app_context():
        feedbacks = Feedback.query.filter(Feedback.corrected_label.isnot(None)).all()

        if len(feedbacks) < RETRAIN_THRESHOLD:
            print(f"Only {len(feedbacks)} corrected samples, need {RETRAIN_THRESHOLD}. Skipping retrain.")
            return False

        print(f"Found {len(feedbacks)} corrected samples")

        data = []
        for fb in feedbacks:
            analysis = MediaAnalysis.query.get(fb.analysis_id)
            if analysis and analysis.media_type == 'image':
                img_path = Path(analysis.file_path)
                if img_path.exists():
                    label = 'ai' if fb.corrected_label else 'human'
                    data.append({'img_name': str(img_path), 'tag': label})

        if len(data) < 10:
            print(f"Only {len(data)} valid images found. Need at least 10 for retraining.")
            return False

        df = pd.DataFrame(data)
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['tag']
        )

        train_df.to_csv('retrain_train.csv', index=False)
        test_df.to_csv('retrain_test.csv', index=False)

        print(f"Created retraining datasets: Train={len(train_df)}, Test={len(test_df)}")
        return True


def fine_tune_model():
    """Fine-tune the existing model with new corrected data."""
    print("Starting fine-tuning process...")
    start_time = time.time()

    from hybrid_train import HybridTrainer

    class FineTuneTrainer(HybridTrainer):
        def create_dataset_csv(self):
            print("Using retraining datasets...")
            if not Path("retrain_train.csv").exists() or not Path("retrain_test.csv").exists():
                return False
            import shutil
            shutil.copy("retrain_train.csv", "train.csv")
            shutil.copy("retrain_test.csv", "test.csv")
            return True

    print("Starting fine-tuning with new data...")

    keras.backend.clear_session()
    gc.collect()

    trainer = FineTuneTrainer()
    trainer.train()

    elapsed = time.time() - start_time
    print(f"Fine-tuning completed in {elapsed:.2f}s")
    return True


def backup_original_model():
    """Backup the current model before retraining."""
    import shutil
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = MODELS_DIR / f"backup_{timestamp}"
    backup_dir.mkdir(exist_ok=True)

    model_files = ['image_classifier.keras', 'model_config.json', 'label_vocabulary.json']
    for file in model_files:
        src = MODELS_DIR / file
        if src.exists():
            shutil.copy2(src, backup_dir / file)

    print(f"Original model backed up to {backup_dir}")
    return backup_dir


def retrain():
    """Main retraining function."""
    print("=" * 60)
    print("RETRAINING PIPELINE TRIGGERED")
    print("=" * 60)

    if not collect_corrected_dataset():
        print("Retraining aborted: insufficient corrected data")
        return False

    backup_dir = backup_original_model()

    try:
        success = fine_tune_model()
        if success:
            print("Model successfully retrained and saved!")
            return True
        else:
            print("Fine-tuning failed")
            return False
    except Exception as e:
        print(f"Retraining error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    retrain()