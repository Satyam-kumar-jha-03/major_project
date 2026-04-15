import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import os
import json
import time
import gc
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate
from PIL import Image

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU available: {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU error: {e}")

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 128
EPOCHS = 50
MAX_SEQ_LENGTH = 1
NUM_FEATURES = 2048

# Thresholds
SMALL_DATASET_THRESHOLD = 5000
MEDIUM_DATASET_THRESHOLD = 20000
VIDEO_MODE_THRESHOLD = 100

class HybridTrainer:
    def __init__(self):
        self.model_type = None
        self.feature_extractor = None
        self.label_processor = None
        self.models_dir = Path(__file__).parent.parent / "models"
        self.models_dir.mkdir(exist_ok=True)

    def analyze_dataset(self, train_df):
        num_samples = len(train_df)
        num_classes = train_df['tag'].nunique()
        class_counts = train_df['tag'].value_counts()
        
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        is_video_frames = any('frame' in name.lower() or 'video' in name.lower() 
                              for name in train_df['img_name'].iloc[:100])
        
        print("\n" + "=" * 60)
        print("📊 DATASET ANALYSIS")
        print("=" * 60)
        print(f"Total samples: {num_samples}")
        print(f"Number of classes: {num_classes}")
        print(f"Class distribution: {dict(class_counts)}")
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        print(f"Video frames detected: {is_video_frames}")
        print("-" * 60)
        
        if is_video_frames and num_samples < VIDEO_MODE_THRESHOLD * num_classes:
            self.model_type = 'video_sequence'
            print("📽️ SELECTED: VIDEO SEQUENCE model")
        elif num_samples < SMALL_DATASET_THRESHOLD:
            self.model_type = 'small_dataset'
            print("🐣 SELECTED: SMALL DATASET model (heavy regularization)")
        elif num_samples < MEDIUM_DATASET_THRESHOLD:
            self.model_type = 'medium_dataset'
            print("📏 SELECTED: MEDIUM DATASET model")
        else:
            self.model_type = 'large_dataset'
            print("🐘 SELECTED: LARGE DATASET model")
        
        global BATCH_SIZE
        if num_samples < 1000:
            BATCH_SIZE = 32
        elif num_samples < 5000:
            BATCH_SIZE = 64
        elif num_samples < 20000:
            BATCH_SIZE = 128
        else:
            BATCH_SIZE = 256
        print(f"Adjusted batch size: {BATCH_SIZE}")
        print("=" * 60)
        
        return {
            'num_samples': num_samples,
            'num_classes': num_classes,
            'imbalance_ratio': imbalance_ratio,
            'is_video_frames': is_video_frames
        }

    def build_feature_extractor(self):
        print(f"🔨 Building feature extractor for {self.model_type}...")
        
        if self.model_type in ['small_dataset', 'video_sequence']:
            feature_extractor = keras.applications.InceptionV3(
                weights="imagenet", include_top=False, pooling="avg",
                input_shape=(IMG_SIZE, IMG_SIZE, 3)
            )
            self.preprocess_input = keras.applications.inception_v3.preprocess_input
            
            if self.model_type == 'small_dataset':
                feature_extractor.trainable = True
                for layer in feature_extractor.layers[:100]:
                    layer.trainable = False
                print(" 🔓 Fine-tuning enabled")
            else:
                feature_extractor.trainable = False
        else:
            feature_extractor = keras.applications.ResNet50(
                weights="imagenet", include_top=False, pooling="avg",
                input_shape=(IMG_SIZE, IMG_SIZE, 3)
            )
            self.preprocess_input = keras.applications.resnet50.preprocess_input
            feature_extractor.trainable = False
        
        return feature_extractor

    def load_and_preprocess_image(self, image_path, augment=False):
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize((IMG_SIZE, IMG_SIZE))
            image_array = np.array(image, dtype=np.float32)
            
            # Data augmentation for small datasets
            if augment and self.model_type == 'small_dataset':
                # Horizontal flip
                if np.random.random() > 0.5:
                    image_array = np.fliplr(image_array)
                
                # Brightness adjustment
                if np.random.random() > 0.7:
                    brightness = np.random.uniform(0.7, 1.3)
                    image_array = image_array * brightness
                    image_array = np.clip(image_array, 0, 255)
                
                # Rotation
                if np.random.random() > 0.7:
                    angle = np.random.uniform(-15, 15)
                    image_array = rotate(image_array, angle, reshape=False, mode='nearest')
                
                # Zoom
                if np.random.random() > 0.7:
                    zoom = np.random.uniform(0.8, 1.2)
                    h, w = image_array.shape[:2]
                    new_h, new_w = int(h * zoom), int(w * zoom)
                    start_h = (h - new_h) // 2
                    start_w = (w - new_w) // 2
                    image_array = image_array[start_h:start_h+new_h, start_w:start_w+new_w]
                    image_array = np.array(Image.fromarray(image_array.astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE)))
            
            image_array = self.preprocess_input(image_array)
            return image_array
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

    def prepare_features(self, df, label_processor, augment=False):
        num_samples = len(df)
        image_paths = df["img_name"].values.tolist()
        labels = df["tag"].values
        labels = label_processor(labels[..., None]).numpy()
        
        frame_features = np.zeros(shape=(num_samples, NUM_FEATURES), dtype="float32")
        successful_samples = 0
        
        self.feature_extractor = self.build_feature_extractor()
        
        batch_size = 32
        total_batches = (num_samples + batch_size - 1) // batch_size
        start_time = time.time()
        
        print(f"Processing {num_samples} images with augmentation={augment}...")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_paths = image_paths[start_idx:end_idx]
            
            batch_images = []
            valid_indices = []
            
            for i, path in enumerate(batch_paths):
                image = self.load_and_preprocess_image(path, augment)
                if image is not None:
                    batch_images.append(image)
                    valid_indices.append(start_idx + i)
            
            if not batch_images:
                continue
            
            batch_images = np.array(batch_images)
            
            try:
                batch_features = self.feature_extractor.predict(batch_images, verbose=0)
                for i, features in enumerate(batch_features):
                    if successful_samples < num_samples:
                        frame_features[successful_samples] = features
                        labels[successful_samples] = labels[valid_indices[i]]
                        successful_samples += 1
            except Exception as e:
                print(f"Error in batch {batch_idx + 1}: {e}")
                continue
            
            if (batch_idx + 1) % 20 == 0:
                print(f" Processed batch {batch_idx + 1}/{total_batches} ({successful_samples}/{num_samples} images)")
        
        frame_features = frame_features[:successful_samples]
        labels = labels[:successful_samples]
        
        elapsed = time.time() - start_time
        print(f"✅ Feature extraction complete: {successful_samples}/{num_samples} images in {elapsed:.2f}s")
        
        return frame_features, labels

    def build_model(self, num_classes, dataset_info):
        print(f"🏗️ Building {self.model_type} classifier...")
        
        if self.model_type == 'video_sequence':
            frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
            mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")
            x = keras.layers.LSTM(128, return_sequences=True)(frame_features_input, mask=mask_input)
            x = keras.layers.Dropout(0.3)(x)
            x = keras.layers.LSTM(64)(x)
            x = keras.layers.Dropout(0.3)(x)
            x = keras.layers.Dense(128, activation="relu")(x)
            x = keras.layers.Dropout(0.4)(x)
            output = keras.layers.Dense(num_classes, activation="softmax")(x)
            model = keras.Model([frame_features_input, mask_input], output)
        
        elif self.model_type == 'small_dataset':
            inputs = keras.Input(shape=(NUM_FEATURES,))
            x = keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.6)(x)
            x = keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(x)
            x = keras.layers.Dropout(0.4)(x)
            output = keras.layers.Dense(num_classes, activation="softmax")(x)
            model = keras.Model(inputs, output)
        
        elif self.model_type == 'medium_dataset':
            inputs = keras.Input(shape=(NUM_FEATURES,))
            x = keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.005))(inputs)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.005))(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.4)(x)
            x = keras.layers.Dense(128, activation="relu")(x)
            x = keras.layers.Dropout(0.3)(x)
            output = keras.layers.Dense(num_classes, activation="softmax")(x)
            model = keras.Model(inputs, output)
        
        else:  # large_dataset
            inputs = keras.Input(shape=(NUM_FEATURES,))
            x = keras.layers.Dense(1024, activation="relu")(inputs)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.4)(x)
            x = keras.layers.Dense(512, activation="relu")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.3)(x)
            x = keras.layers.Dense(256, activation="relu")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.2)(x)
            x = keras.layers.Dense(128, activation="relu")(x)
            output = keras.layers.Dense(num_classes, activation="softmax")(x)
            model = keras.Model(inputs, output)
        
        lr = 0.0005 if dataset_info['num_samples'] < 1000 else (0.001 if dataset_info['num_samples'] < 5000 else 0.001)
        
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            metrics=["accuracy"]
        )
        print(f" Learning rate: {lr}")
        
        return model

    def create_dataset_csv(self):
        print("\n📁 Creating dataset CSV files...")
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        dataset_path = project_root / "dataset"
        train_path = dataset_path / "train"
        test_path = dataset_path / "test"
        
        if not train_path.exists():
            print(f"❌ Training path {train_path} does not exist!")
            print("Please create: dataset/train/fake/ and dataset/train/real/")
            return False
        
        # Process training data
        train_rooms = []
        label_types = [d for d in os.listdir(train_path) if os.path.isdir(train_path / d)]
        
        if not label_types:
            print(f"❌ No class folders found in {train_path}!")
            return False
        
        print(f"📁 Found class folders: {label_types}")
        
        for item in label_types:
            class_path = train_path / item
            if os.path.isdir(class_path):
                all_images = set()
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                    for file in class_path.glob(ext):
                        all_images.add(file.name)
                    for file in class_path.glob(ext.upper()):
                        all_images.add(file.name)
                print(f" 📸 {item}: found {len(all_images)} images")
                for image in all_images:
                    train_rooms.append((item, str(class_path / image)))
        
        if not train_rooms:
            print("❌ No training images found!")
            return False
        
        train_df = pd.DataFrame(data=train_rooms, columns=['tag', 'img_name'])
        train_df = train_df.loc[:, ['img_name', 'tag']]
        train_df.to_csv('train.csv', index=False)
        print(f"✅ Created train.csv with {len(train_df)} samples")
        
        # Process test data
        test_rooms = []
        if test_path.exists():
            for item in label_types:
                class_path = test_path / item
                if os.path.isdir(class_path):
                    all_images = set()
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                        for file in class_path.glob(ext):
                            all_images.add(file.name)
                        for file in class_path.glob(ext.upper()):
                            all_images.add(file.name)
                    for image in all_images:
                        test_rooms.append((item, str(class_path / image)))
            
            if test_rooms:
                test_df = pd.DataFrame(data=test_rooms, columns=['tag', 'img_name'])
                test_df = test_df.loc[:, ['img_name', 'tag']]
                test_df.to_csv('test.csv', index=False)
                print(f"✅ Created test.csv with {len(test_df)} samples")
            else:
                # Create test split from training data
                train_df, test_df = train_test_split(
                    train_df, test_size=0.2, random_state=42, stratify=train_df['tag']
                )
                train_df.to_csv('train.csv', index=False)
                test_df.to_csv('test.csv', index=False)
                print(f"✅ Created train/test split: {len(train_df)}/{len(test_df)}")
        else:
            # Create test split from training data
            train_df, test_df = train_test_split(
                train_df, test_size=0.2, random_state=42, stratify=train_df['tag']
            )
            train_df.to_csv('train.csv', index=False)
            test_df.to_csv('test.csv', index=False)
            print(f"✅ Created train/test split: {len(train_df)}/{len(test_df)}")
        
        return True

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title(f'Model Accuracy ({self.model_type})', fontsize=12)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title(f'Model Loss ({self.model_type})', fontsize=12)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.models_dir / 'training_history.png', dpi=150)
        plt.show()
        print(f"📈 Training history saved")

    def train(self):
        print("=" * 60)
        print("🚀 STARTING HYBRID TRAINING SYSTEM")
        print("=" * 60)
        start_time = time.time()
        
        if not self.create_dataset_csv():
            print("❌ Failed to create dataset CSVs.")
            return
        
        train_df = pd.read_csv("train.csv")
        test_df = pd.read_csv("test.csv")
        print(f"\n📊 Dataset sizes: Train={len(train_df)}, Test={len(test_df)}")
        
        dataset_info = self.analyze_dataset(train_df)
        
        self.label_processor = keras.layers.StringLookup(
            num_oov_indices=0, vocabulary=np.unique(train_df["tag"])
        )
        vocab = self.label_processor.get_vocabulary()
        print(f"\n🏷️ Classes: {vocab}")
        
        print("\n🔄 Preparing training features...")
        augment_training = (self.model_type == 'small_dataset')
        train_features, train_labels = self.prepare_features(
            train_df, self.label_processor, augment=augment_training
        )
        
        print("\n🔄 Preparing test features...")
        test_features, test_labels = self.prepare_features(
            test_df, self.label_processor, augment=False
        )
        
        print("\n🏗️ Building model...")
        model = self.build_model(len(vocab), dataset_info)
        print(model.summary())
        
        if self.model_type == 'video_sequence':
            train_data = [train_features[:, np.newaxis, :], np.ones((len(train_features), MAX_SEQ_LENGTH), dtype="bool")]
            test_data = [test_features[:, np.newaxis, :], np.ones((len(test_features), MAX_SEQ_LENGTH), dtype="bool")]
        else:
            train_data = train_features
            test_data = test_features
        
        model_path = self.models_dir / "image_classifier.keras"
        
        checkpoint = keras.callbacks.ModelCheckpoint(
            str(model_path), save_best_only=True, monitor='val_accuracy', mode='max', verbose=1
        )
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5, restore_best_weights=True, mode='max', verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=0.000001, verbose=1
        )
        
        print("\n🎯 Starting training...")
        print(f" Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
        print(f" Early stopping patience: 5")
        print("-" * 60)
        
        history = model.fit(
            train_data, train_labels,
            validation_data=(test_data, test_labels),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            verbose=1
        )
        
        print("\n📈 Evaluating model...")
        test_loss, test_accuracy = model.evaluate(test_data, test_labels, verbose=0)
        print(f"✅ Test accuracy: {test_accuracy:.4f}")
        print(f"✅ Test loss: {test_loss:.4f}")
        
        self.plot_training_history(history)
        
        with open(self.models_dir / 'label_vocabulary.json', 'w') as f:
            json.dump([str(v) for v in vocab], f)
        
        config = {
            'img_size': IMG_SIZE,
            'num_features': NUM_FEATURES,
            'batch_size': BATCH_SIZE,
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'class_names': [str(v) for v in vocab],
            'model_type': self.model_type,
            'dataset_info': dataset_info,
            'total_training_time': time.time() - start_time,
            'augmentation_used': augment_training,
            'early_stopping_patience': 5
        }
        
        with open(self.models_dir / 'model_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"💾 Model saved to: {model_path}")
        print(f"📊 Test Accuracy: {test_accuracy:.2%}")
        print(f"🤖 Model Type: {self.model_type}")
        print(f"⏱️ Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print("=" * 60)

def main():
    keras.backend.clear_session()
    gc.collect()
    trainer = HybridTrainer()
    trainer.train()

if __name__ == "__main__":
    main()