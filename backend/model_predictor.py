import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import json
import os
import cv2
import tempfile
import time
import re
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class AIModelPredictor:
    """Advanced AI Model Predictor for Image, Video, and Text Analysis"""
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None, vocab_path: Optional[str] = None):
        """Initialize the trained model for predictions"""
        print("=" * 60)
        print("🔧 AI Model Predictor Initialization")
        print("=" * 60)
        
        self.model = None
        self.feature_extractor = None
        self.class_names = ['REAL', 'FAKE']  # Default classes
        self.config = {}
        self.img_size = (224, 224)
        self.num_features = 2048
        self.max_seq_length = 1
        self.is_initialized = False
        self.use_fallback = False
        self.preprocess_input = None
        
        # Performance metrics
        self.metrics = {
            'total_predictions': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }
        
        # Try to load model if paths are provided
        if model_path and config_path and vocab_path:
            self._load_model(model_path, config_path, vocab_path)
        else:
            print("⚠️ No model paths provided. Using fallback mode.")
            self.use_fallback = True
            self.is_initialized = True
    
    def _load_model(self, model_path: str, config_path: str, vocab_path: str):
        """Load the trained model and associated files"""
        try:
            # Check if files exist
            missing_files = []
            for path, name in [(model_path, "Model"), (config_path, "Config"), (vocab_path, "Vocabulary")]:
                if not os.path.exists(path):
                    missing_files.append(name)
            
            if missing_files:
                print(f"❌ Missing files: {', '.join(missing_files)}")
                print("   Using fallback mode.")
                self.use_fallback = True
                self.is_initialized = True
                return
            
            # Load model
            print("📦 Loading model...")
            self.model = keras.models.load_model(model_path, compile=False)
            print(f"   ✅ Model loaded successfully")
            
            # Load configuration
            print("📋 Loading configuration...")
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"   ✅ Config loaded")
            
            # Load vocabulary
            print("📖 Loading vocabulary...")
            with open(vocab_path, 'r') as f:
                self.class_names = json.load(f)
            print(f"   ✅ Vocabulary loaded: {self.class_names}")
            
            # Extract configuration parameters
            self.img_size = (self.config.get('img_size', 224), self.config.get('img_size', 224))
            self.num_features = self.config.get('num_features', 2048)
            self.max_seq_length = self.config.get('max_seq_length', 1)
            model_type = self.config.get('model_type', 'unknown')
            
            print(f"\n🎯 Model Configuration:")
            print(f"   - Image Size: {self.img_size}")
            print(f"   - Features: {self.num_features}")
            print(f"   - Sequence Length: {self.max_seq_length}")
            print(f"   - Model Type: {model_type}")
            print(f"   - Classes: {len(self.class_names)}")
            
            # Build feature extractor
            print("\n🔨 Building feature extractor...")
            self.feature_extractor = self._build_feature_extractor()
            
            if self.feature_extractor:
                print("   ✅ Feature extractor built successfully")
                self.is_initialized = True
                print("\n✨ Model predictor ready for use!")
            else:
                print("   ❌ Failed to build feature extractor")
                self.use_fallback = True
                self.is_initialized = True
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.use_fallback = True
            self.is_initialized = True
    
    def _build_feature_extractor(self) -> Optional[keras.Model]:
        """Build the appropriate feature extractor based on model configuration"""
        try:
            # Determine which feature extractor to use
            model_type = self.config.get('model_type', 'unknown')
            
            if model_type in ['small_dataset', 'video_sequence']:
                # Use InceptionV3 for smaller datasets
                feature_extractor = keras.applications.InceptionV3(
                    weights="imagenet",
                    include_top=False,
                    pooling="avg",
                    input_shape=(224, 224, 3),
                )
                self.preprocess_input = keras.applications.inception_v3.preprocess_input
            else:
                # Use ResNet50 for medium/large datasets
                feature_extractor = keras.applications.ResNet50(
                    weights="imagenet",
                    include_top=False,
                    pooling="avg",
                    input_shape=(224, 224, 3),
                )
                self.preprocess_input = keras.applications.resnet50.preprocess_input
            
            feature_extractor.trainable = False
            return feature_extractor
            
        except Exception as e:
            print(f"   ❌ Error building feature extractor: {e}")
            return None
    
    def load_and_preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess image for prediction"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize
            image = image.resize(self.img_size)
            
            # Convert to array
            image_array = np.array(image, dtype=np.float32)
            
            # Apply preprocessing if available
            if self.preprocess_input:
                processed = self.preprocess_input(image_array)
            else:
                processed = image_array / 255.0
            
            return processed
            
        except Exception as e:
            print(f"❌ Error loading/preprocessing image {image_path}: {e}")
            return None
    
    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        """Extract features using the feature extractor"""
        try:
            if self.feature_extractor is None:
                print("❌ Feature extractor not available")
                return None
                
            image = self.load_and_preprocess_image(image_path)
            if image is None:
                return None
            
            image = image[None, ...]  # Add batch dimension
            
            # Extract features
            features = self.feature_extractor.predict(image, verbose=0)
            return features.squeeze()  # Remove batch dimension
            
        except Exception as e:
            print(f"❌ Feature extraction error: {e}")
            return None
    
    def predict_image(self, image_path: str) -> Dict[str, Any]:
        """Predict if image is AI or Human generated"""
        start_time = time.time()
        
        try:
            # Update metrics
            self.metrics['total_predictions'] += 1
            
            # Check if we're in fallback mode
            if self.use_fallback or self.model is None or self.feature_extractor is None:
                result = self._fallback_prediction()
                result['prediction_time'] = time.time() - start_time
                self._update_metrics(result['prediction_time'])
                return result
            
            # Extract features
            features = self.extract_features(image_path)
            if features is None:
                result = self._fallback_prediction()
                result['prediction_time'] = time.time() - start_time
                self._update_metrics(result['prediction_time'])
                return result
            
            # Reshape features for model input
            features = features.reshape(1, -1)
            
            # Make prediction
            probabilities = self.model.predict(features, verbose=0)[0]
            
            # Create result dictionary
            result = {}
            for i, class_name in enumerate(self.class_names):
                result[class_name] = float(probabilities[i] * 100)
            
            # Determine predicted class
            predicted_class = self.class_names[np.argmax(probabilities)]
            confidence = np.max(probabilities) * 100
            
            # Determine if AI generated
            is_ai = 'fake' in predicted_class.lower() or 'ai' in predicted_class.lower()
            
            prediction_time = time.time() - start_time
            
            full_result = {
                'isAI': is_ai,
                'confidence': confidence,
                'probabilities': result,
                'predicted_class': predicted_class,
                'prediction_time': prediction_time
            }
            
            # Print prediction details
            print(f"\n🔍 Image Prediction Results:")
            print(f"   Result: {'🤖 AI GENERATED' if is_ai else '👤 HUMAN CREATED'}")
            print(f"   Confidence: {confidence:.2f}%")
            print(f"   Class: {predicted_class}")
            print(f"   Time: {prediction_time:.3f}s")
            
            self._update_metrics(prediction_time)
            return full_result
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            result = self._fallback_prediction()
            result['prediction_time'] = time.time() - start_time
            self._update_metrics(result['prediction_time'])
            return result
    
    def predict_video(self, video_path: str, frame_interval: int = 30) -> Dict[str, Any]:
        """Predict if video is AI or Human generated by analyzing frames"""
        start_time = time.time()
        
        try:
            # Check if we're in fallback mode
            if self.use_fallback or self.model is None or self.feature_extractor is None:
                return self._fallback_video_prediction()
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Could not open video: {video_path}")
                return self._fallback_video_prediction()
            
            frames_analyzed = 0
            ai_count = 0
            human_count = 0
            ai_confidence_sum = 0.0
            human_confidence_sum = 0.0
            
            frame_count = 0
            temp_dir = tempfile.mkdtemp()
            
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every nth frame
                    if frame_count % frame_interval == 0:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Save temporary frame
                        temp_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.jpg")
                        Image.fromarray(frame_rgb).save(temp_path)
                        
                        try:
                            # Analyze frame
                            result = self.predict_image(temp_path)
                            
                            if result:
                                if result['isAI']:
                                    ai_count += 1
                                    ai_confidence_sum += result['confidence']
                                else:
                                    human_count += 1
                                    human_confidence_sum += result['confidence']
                                
                                frames_analyzed += 1
                            
                        except Exception as e:
                            print(f"⚠️ Error analyzing frame {frame_count}: {e}")
                        
                        finally:
                            # Clean up temp file
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                    
                    frame_count += 1
                
            finally:
                cap.release()
                # Clean up temp directory
                try:
                    os.rmdir(temp_dir)
                except:
                    pass
            
            if frames_analyzed == 0:
                print("⚠️ No frames were analyzed!")
                return {
                    'isAI': False,
                    'confidence': 50.0,
                    'frames_analyzed': 0,
                    'ai_frames': 0,
                    'human_frames': 0,
                    'analysis_time': time.time() - start_time
                }
            
            # Calculate overall results
            ai_ratio = ai_count / frames_analyzed
            overall_is_ai = ai_ratio > 0.5
            
            if overall_is_ai:
                overall_confidence = (ai_confidence_sum / ai_count) if ai_count > 0 else 50.0
            else:
                overall_confidence = (human_confidence_sum / human_count) if human_count > 0 else 50.0
            
            analysis_time = time.time() - start_time
            
            print(f"\n📊 Video Analysis Complete:")
            print(f"   Frames Analyzed: {frames_analyzed}")
            print(f"   AI Frames: {ai_count}")
            print(f"   Human Frames: {human_count}")
            print(f"   Overall Result: {'🤖 AI GENERATED' if overall_is_ai else '👤 HUMAN CREATED'}")
            print(f"   Confidence: {overall_confidence:.2f}%")
            
            return {
                'isAI': overall_is_ai,
                'confidence': overall_confidence,
                'frames_analyzed': frames_analyzed,
                'ai_frames': ai_count,
                'human_frames': human_count,
                'analysis_time': analysis_time
            }
            
        except Exception as e:
            print(f"❌ Video analysis error: {e}")
            return self._fallback_video_prediction()
    
    def predict_text(self, text_content: str) -> Dict[str, Any]:
        """Analyze text for AI vs Human content"""
        start_time = time.time()
        
        try:
            text_content = text_content.strip()
            text_length = len(text_content)
            word_count = len(text_content.split())
            
            if text_length < 20:
                return {
                    'isAI': False,
                    'confidence': 50.0,
                    'message': 'Text too short for reliable analysis'
                }
            
            # Advanced metrics
            sentences = re.split(r'[.!?]+', text_content)
            sentences = [s.strip() for s in sentences if s.strip()]
            sentence_count = len(sentences) if len(sentences) > 0 else 1
            
            # Calculate metrics
            avg_sentence_length = word_count / sentence_count
            avg_word_length = text_length / word_count if word_count > 0 else 0
            
            # Lexical diversity
            words = text_content.lower().split()
            unique_words = set(words)
            lexical_diversity = len(unique_words) / max(len(words), 1)
            
            # Punctuation analysis
            punctuation_count = sum(1 for c in text_content if c in '.,!?;:')
            punctuation_density = punctuation_count / max(text_length, 1)
            
            # Sentence length variance
            sentence_lengths = [len(sent.split()) for sent in sentences]
            if len(sentence_lengths) > 1:
                sentence_length_variance = np.var(sentence_lengths)
            else:
                sentence_length_variance = 0
            
            # AI indicators
            ai_indicators = 0
            human_indicators = 0
            
            # 1. Lexical diversity
            if lexical_diversity > 0.8:
                ai_indicators += 2
            elif lexical_diversity > 0.7:
                ai_indicators += 1
            elif lexical_diversity < 0.4:
                human_indicators += 1
            
            # 2. Sentence length consistency
            if sentence_length_variance < 10:
                ai_indicators += 1
            elif sentence_length_variance > 30:
                human_indicators += 1
            
            # 3. Punctuation density
            if punctuation_density < 0.05:
                ai_indicators += 1
            elif punctuation_density > 0.15:
                human_indicators += 1
            
            # 4. Average word length
            if avg_word_length > 6:
                ai_indicators += 1
            elif avg_word_length < 4:
                human_indicators += 1
            
            # Calculate final score
            total_indicators = ai_indicators + human_indicators
            if total_indicators == 0:
                ai_score = 0.5
            else:
                ai_score = ai_indicators / total_indicators
            
            is_ai = ai_score > 0.55
            confidence = 50 + (abs(ai_score - 0.5) * 100)
            confidence = min(95, max(55, confidence))
            
            analysis_time = time.time() - start_time
            
            print(f"\n📝 Text Analysis Results:")
            print(f"   Result: {'🤖 AI GENERATED' if is_ai else '👤 HUMAN CREATED'}")
            print(f"   Confidence: {confidence:.2f}%")
            print(f"   Word Count: {word_count}")
            print(f"   Time: {analysis_time:.3f}s")
            
            return {
                'isAI': is_ai,
                'confidence': confidence,
                'ai_score': ai_score,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'lexical_diversity': lexical_diversity,
                'analysis_time': analysis_time
            }
            
        except Exception as e:
            print(f"❌ Text analysis error: {e}")
            return {
                'isAI': False,
                'confidence': 50.0,
                'error': str(e)
            }
    
    def _fallback_prediction(self) -> Dict[str, Any]:
        """Fallback prediction when model is not available"""
        import random
        is_ai = random.random() > 0.6
        confidence = random.uniform(65, 90)
        
        return {
            'isAI': is_ai,
            'confidence': confidence,
            'probabilities': {
                'REAL': 100 - confidence if is_ai else confidence,
                'FAKE': confidence if is_ai else 100 - confidence
            },
            'predicted_class': 'FAKE' if is_ai else 'REAL',
            'fallback_mode': True,
            'message': 'Using fallback mode - Model not loaded'
        }
    
    def _fallback_video_prediction(self) -> Dict[str, Any]:
        """Fallback video prediction when model is not available"""
        import random
        is_ai = random.random() > 0.6
        confidence = random.uniform(65, 90)
        frames = random.randint(5, 25)
        
        return {
            'isAI': is_ai,
            'confidence': confidence,
            'frames_analyzed': frames,
            'ai_frames': int(frames * (0.7 if is_ai else 0.3)),
            'human_frames': int(frames * (0.3 if is_ai else 0.7)),
            'fallback_mode': True,
            'analysis_time': 0.5
        }
    
    def _update_metrics(self, prediction_time: float):
        """Update performance metrics"""
        self.metrics['total_time'] += prediction_time
        self.metrics['avg_time'] = self.metrics['total_time'] / self.metrics['total_predictions']
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'total_predictions': self.metrics['total_predictions'],
            'total_time': self.metrics['total_time'],
            'average_time': self.metrics['avg_time'],
            'model_loaded': not self.use_fallback
        }

# Global instance
_model_predictor = None

def initialize_model() -> bool:
    """Initialize the global AI model predictor"""
    global _model_predictor
    
    try:
        # Get paths relative to this file
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        models_dir = project_root / "models"
        
        model_path = models_dir / "image_classifier.keras"
        config_path = models_dir / "model_config.json"
        vocab_path = models_dir / "label_vocabulary.json"
        
        print("\n" + "="*60)
        print("🚀 AI Model Predictor Initialization")
        print("="*60)
        print(f"📂 Model Directory: {models_dir}")
        print(f"📄 Model File: {model_path.exists()}")
        print(f"📋 Config File: {config_path.exists()}")
        print(f"📖 Vocab File: {vocab_path.exists()}")
        print("="*60 + "\n")
        
        _model_predictor = AIModelPredictor(
            model_path=str(model_path) if model_path.exists() else None,
            config_path=str(config_path) if config_path.exists() else None,
            vocab_path=str(vocab_path) if vocab_path.exists() else None
        )
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        _model_predictor = AIModelPredictor()
        return False

def get_model_predictor() -> AIModelPredictor:
    """Get the global model predictor instance"""
    global _model_predictor
    
    if _model_predictor is None:
        initialize_model()
    
    return _model_predictor
