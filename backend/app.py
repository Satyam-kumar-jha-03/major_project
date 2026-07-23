"""
REALYTICS - AI vs Human Media & Text Detector
Complete Flask Application with Authentication, Free Trial, and AI Model
"""
from flask import Flask, request, jsonify, send_from_directory, session, send_file
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import os
import uuid
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import re
import json
from pathlib import Path
import random
import tempfile
import time
import io
from functools import wraps
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

# TensorFlow / Keras imports
import tensorflow as tf
import keras
from keras import layers, models

# Import configuration
from backend.config import (
    FRONTEND_DIR, UPLOAD_FOLDER, DATABASE_URI, MAX_CONTENT_LENGTH,
    ALLOWED_IMAGE_EXTS, ALLOWED_VIDEO_EXTS, DEMO_MODE,
    GEMINI_API_KEY, GEMINI_MODEL, ENABLE_EXPLANATIONS,
    MAX_VIDEO_DURATION_SECONDS, FREE_TRIAL_LIMIT, JWT_SECRET_KEY,
    RETRAIN_THRESHOLD, MODELS_DIR
)
from backend.models import db, User, MediaAnalysis, Feedback
from backend.auth import login_required

# Optional Gemini import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    if ENABLE_EXPLANATIONS and GEMINI_API_KEY and GEMINI_API_KEY != '':
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ Gemini API configured")
    else:
        print("⚠️ Gemini explanations disabled")
        GEMINI_AVAILABLE = False
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ google-generativeai not installed")

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================
app = Flask(__name__)
CORS(app, supports_credentials=True)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = JWT_SECRET_KEY
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

UPLOAD_FOLDER.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

db.init_app(app)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_file_extension(filename):
    """Simple file extension validation"""
    ext = os.path.splitext(filename)[1].lower()
    if ext in ALLOWED_IMAGE_EXTS:
        return True, 'image', ext
    elif ext in ALLOWED_VIDEO_EXTS:
        return True, 'video', ext
    return False, None, None


def generate_simple_explanation(media_type, is_ai, confidence):
    """Generate a simple explanation without Gemini"""
    if is_ai:
        return f"This {media_type} shows patterns consistent with AI generation. The system is {confidence:.0f}% confident in this assessment. Key indicators include texture patterns and structural consistency typical of AI models."
    else:
        return f"This {media_type} appears to be human-created. The system is {confidence:.0f}% confident in this assessment. The content shows natural variation and organic patterns typical of human creation."


def generate_gemini_explanation(media_type, is_ai, confidence, text_content=None):
    """Generate explanation using Gemini API"""
    if not ENABLE_EXPLANATIONS or not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return generate_simple_explanation(media_type, is_ai, confidence)
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        if media_type == 'image':
            prompt = f"""An image was classified as {'AI-GENERATED' if is_ai else 'HUMAN-CREATED'} with {confidence:.1f}% confidence.
Explain in 2-3 simple sentences why. Focus on visual patterns like textures, edges, or lighting. Be specific and helpful."""
        elif media_type == 'video':
            prompt = f"""A video was classified as {'AI-GENERATED' if is_ai else 'REAL'} with {confidence:.1f}% confidence.
Explain in 2-3 simple sentences why."""
        elif media_type == 'text':
            preview = text_content[:200] if text_content else ""
            prompt = f"""Text was classified as {'AI-GENERATED' if is_ai else 'HUMAN-WRITTEN'} with {confidence:.1f}% confidence.
Text preview: "{preview}"
Explain in 2-3 simple sentences why. Focus on sentence patterns, word choice, or flow. Be conversational."""
        else:
            return generate_simple_explanation(media_type, is_ai, confidence)
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return generate_simple_explanation(media_type, is_ai, confidence)


def analyze_text_simple(text_content):
    """Simple text analysis"""
    if DEMO_MODE:
        return random.random() > 0.6, random.uniform(70, 95)
    
    text_length = len(text_content)
    if text_length < 50:
        return False, 50.0
    
    words = text_content.lower().split()
    unique_words = set(words)
    lexical_diversity = len(unique_words) / max(len(words), 1)
    
    sentences = re.split(r'[.!?]+', text_content)
    sentence_lengths = [len(sent.split()) for sent in sentences if sent.strip()]
    
    ai_score = 0
    if sentence_lengths:
        variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
        if variance < 5:
            ai_score += 0.3
    if lexical_diversity > 0.7:
        ai_score += 0.3
    
    formal_words = ['utilization', 'methodology', 'optimization', 'furthermore', 'consequently', 'additionally', 'moreover']
    formal_count = sum(1 for w in words if w in formal_words)
    if formal_count > 2:
        ai_score += 0.2
    
    is_ai = ai_score > 0.5
    confidence = 50 + (ai_score * 40) if is_ai else 90 - (ai_score * 40)
    confidence = min(95, max(60, confidence))
    
    return is_ai, float(confidence)


def generate_pdf_report(analysis):
    """Generate PDF report for an analysis"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#d4af37'),
        alignment=TA_CENTER,
        spaceAfter=30
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#d4af37'),
        spaceAfter=12
    )

    result_style = ParagraphStyle(
        'ResultStyle',
        parent=styles['Normal'],
        fontSize=18,
        textColor=colors.HexColor('#4caf50') if not analysis.is_ai else colors.HexColor('#f44336'),
        alignment=TA_CENTER,
        spaceAfter=20
    )

    story = []

    story.append(Paragraph("REALYTICS - Analysis Report", title_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Report ID: {analysis.id}", styles['Normal']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"Analysis Date: {analysis.created_at.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Detection Result", heading_style))
    result_text = f"<b>{'AI-GENERATED' if analysis.is_ai else 'HUMAN-CREATED'}</b>"
    story.append(Paragraph(result_text, result_style))
    story.append(Spacer(1, 10))

    confidence_percent = analysis.confidence
    story.append(Paragraph(f"Confidence: {confidence_percent:.1f}%", styles['Normal']))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Media Information", heading_style))
    story.append(Paragraph(f"Type: {analysis.media_type.upper()}", styles['Normal']))
    story.append(Paragraph(f"Filename: {analysis.filename}", styles['Normal']))
    story.append(Spacer(1, 20))

    if analysis.explanation:
        story.append(Paragraph("AI Explanation", heading_style))
        story.append(Paragraph(analysis.explanation[:500], styles['Normal']))
        story.append(Spacer(1, 20))

    story.append(Paragraph("— Report generated by REALYTICS System —",
                          ParagraphStyle('Footer', parent=styles['Normal'],
                                       fontSize=8, textColor=colors.grey, alignment=TA_CENTER)))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ============================================================================
# AI MODEL PREDICTOR - USING YOUR TRAINED MODEL
# ============================================================================

class AIModelPredictor:
    """AI Model Predictor - Works with your trained model (fake/real classes)"""
    
    def __init__(self):
        print("=" * 60)
        print(" Loading AI Model Predictor...")
        print("=" * 60)

        self.models_dir = MODELS_DIR
        self.model_path = self.models_dir / "image_classifier.keras"
        self.config_path = self.models_dir / "model_config.json"
        self.vocab_path = self.models_dir / "label_vocabulary.json"

        print(f" Model path: {self.model_path}")
        print(f" Config path: {self.config_path}")
        print(f" Vocab path: {self.vocab_path}")

        self.model = None
        self.feature_extractor = None
        self.class_names = []
        self.config = {}
        self.img_size = (224, 224)
        self.num_features = 2048
        self.is_initialized = False
        self.use_fallback = False
        self.demo_mode = DEMO_MODE

        if not self.demo_mode:
            self._load_model()
        else:
            print("🎮 DEMO MODE ENABLED - Using random predictions")
            self.is_initialized = True
            self.use_fallback = True

    def _load_model(self):
        try:
            # Check if model files exist
            if not self.model_path.exists():
                print(f"❌ Model file not found: {self.model_path}")
                self.use_fallback = True
                self.is_initialized = True
                return

            if not self.config_path.exists():
                print(f"⚠️ Config file not found, using defaults")
            
            if not self.vocab_path.exists():
                print(f"⚠️ Vocab file not found, using defaults")

            print("📦 Loading classifier model...")
            self.model = keras.models.load_model(str(self.model_path))
            print("✅ Model loaded successfully")

            # Load config
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                self.img_size = (self.config.get('img_size', 224), self.config.get('img_size', 224))
                print(f"✅ Config loaded - Image size: {self.img_size}")
                if 'test_accuracy' in self.config:
                    print(f"   Test Accuracy: {self.config['test_accuracy']:.2%}")

            # Load vocabulary
            if self.vocab_path.exists():
                with open(self.vocab_path, 'r') as f:
                    self.class_names = json.load(f)
                print(f"✅ Classes: {self.class_names}")
            else:
                self.class_names = ['fake', 'real']
                print(f"✅ Using default classes: {self.class_names}")

            # Build feature extractor
            self._build_feature_extractor()

            self.is_initialized = True
            self.use_fallback = False
            print("🎯 Model is ONLINE and ready!")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.use_fallback = True
            self.is_initialized = True

    def _build_feature_extractor(self):
        try:
            self.feature_extractor = keras.applications.ResNet50(
                weights="imagenet",
                include_top=False,
                pooling="avg",
                input_shape=(224, 224, 3),
            )
            self.feature_extractor.trainable = False
            self.preprocess_fn = keras.applications.resnet50.preprocess_input
            self.num_features = 2048
            print("  ✅ ResNet50 feature extractor ready")
        except Exception as e:
            print(f"  ⚠️ Feature extractor failed: {e}")

    def load_and_preprocess_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize(self.img_size)
            image_array = np.array(image, dtype=np.float32)
            return image_array
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
        
    def extract_features(self, image_array):
        try:
            preprocessed = self.preprocess_fn(image_array.copy())
            batch_input = preprocessed[None, ...]
            features = self.feature_extractor.predict(batch_input, verbose=0)[0]
            return features
        except Exception as e:
            print(f"⚠️ Feature extraction failed: {e}")
            return None

    def predict_image(self, image_path):
        """Predict if image is AI (fake) or Human (real)"""
        start_time = time.time()
        
        if self.demo_mode:
            return self._demo_prediction()
        if self.use_fallback or self.model is None:
            return self._fallback_prediction()

        try:
            image_array = self.load_and_preprocess_image(image_path)
            if image_array is None:
                return self._fallback_prediction()

            features = self.extract_features(image_array)
            if features is None:
                return self._fallback_prediction()

            features = features.reshape(1, -1)
            probabilities = self.model.predict(features, verbose=0)[0]

            # Handle binary output (sigmoid) or multi-class
            if len(probabilities) == 1:
                confidence = float(probabilities[0] * 100)
                is_ai = confidence > 50  # Higher confidence = more likely fake/AI
                predicted_class = 'fake' if is_ai else 'real'
            else:
                result = {}
                for i, class_name in enumerate(self.class_names):
                    result[class_name] = float(probabilities[i] * 100)
                predicted_idx = np.argmax(probabilities)
                predicted_class = self.class_names[predicted_idx]
                confidence = float(probabilities[predicted_idx] * 100)
                is_ai = predicted_class.lower() == 'fake'

            elapsed = time.time() - start_time
            
            return {
                'isAI': is_ai,
                'confidence': confidence,
                'predicted_class': predicted_class,
                'prediction_time': elapsed,
                'model_status': 'online'
            }

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return self._fallback_prediction()

    def predict_video(self, video_path, frame_interval=10, max_frames=200):
        if self.demo_mode:
            return self._demo_video_prediction()
        if self.use_fallback or self.model is None:
            return self._fallback_video_prediction()

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return self._fallback_video_prediction()

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            frames_to_sample = total_frames // frame_interval
            if frames_to_sample > max_frames:
                frame_interval = max(1, total_frames // max_frames)
                print(f" Adjusted sampling: every {frame_interval}th frame")

            print(f"🎬 Video analysis: {duration:.1f}s, sampling every {frame_interval}th frame")

            frame_predictions = []
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    temp_path = os.path.join(tempfile.gettempdir(), f"temp_frame_{frame_count}.jpg")
                    Image.fromarray(frame_rgb).save(temp_path)

                    result = self.predict_image(temp_path)
                    if result and 'error' not in result:
                        frame_predictions.append({
                            'frame': frame_count,
                            'isAI': result['isAI'],
                            'confidence': result['confidence']
                        })

                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                frame_count += 1

            cap.release()

            if len(frame_predictions) == 0:
                return self._fallback_video_prediction()

            ai_count = sum(1 for p in frame_predictions if p['isAI'])
            human_count = len(frame_predictions) - ai_count
            overall_is_ai = ai_count > human_count

            winning_confidences = [p['confidence'] for p in frame_predictions if p['isAI'] == overall_is_ai]
            overall_confidence = sum(winning_confidences) / len(winning_confidences) if winning_confidences else 50.0

            return {
                'isAI': overall_is_ai,
                'confidence': overall_confidence,
                'frames_analyzed': len(frame_predictions),
                'ai_frames': ai_count,
                'human_frames': human_count,
                'video_duration': duration,
                'model_status': 'online'
            }

        except Exception as e:
            print(f"❌ Video analysis error: {e}")
            return self._fallback_video_prediction()

    def _demo_prediction(self):
        is_ai = random.random() > 0.6
        confidence = random.uniform(75.0, 95.0)
        return {
            'isAI': is_ai,
            'confidence': confidence,
            'predicted_class': 'fake' if is_ai else 'real',
            'model_status': 'demo'
        }

    def _demo_video_prediction(self):
        is_ai = random.random() > 0.6
        confidence = random.uniform(75.0, 95.0)
        frames_analyzed = random.randint(5, 20)
        return {
            'isAI': is_ai,
            'confidence': confidence,
            'frames_analyzed': frames_analyzed,
            'ai_frames': random.randint(0, frames_analyzed),
            'human_frames': random.randint(0, frames_analyzed),
            'model_status': 'demo'
        }

    def _fallback_prediction(self):
        return {
            'isAI': False,
            'confidence': 50.0,
            'predicted_class': 'unknown',
            'error': 'Model unavailable',
            'model_status': 'offline'
        }

    def _fallback_video_prediction(self):
        return {
            'isAI': False,
            'confidence': 50.0,
            'frames_analyzed': 0,
            'ai_frames': 0,
            'human_frames': 0,
            'error': 'Model unavailable',
            'model_status': 'offline'
        }

    def get_status(self):
        if self.is_initialized and not self.use_fallback and not self.demo_mode:
            return {
                'status': 'online',
                'classes': self.class_names,
                'accuracy': self.config.get('test_accuracy', 'unknown')
            }
        elif self.demo_mode:
            return {'status': 'demo_mode'}
        else:
            return {'status': 'offline', 'error': 'Model not loaded'}


# Initialize global model predictor
print("\n" + "=" * 60)
print("Initializing AI Model...")
print("=" * 60)

model_predictor = AIModelPredictor()

if model_predictor.is_initialized and not model_predictor.use_fallback and not DEMO_MODE:
    print("✅ AI Model is ONLINE and ready for predictions!")
    print(f"   Classes: {model_predictor.class_names}")
    if model_predictor.config.get('test_accuracy'):
        print(f"   Test Accuracy: {model_predictor.config['test_accuracy']:.2%}")
elif DEMO_MODE:
    print("⚠️ DEMO MODE - Using random predictions")
else:
    print("⚠️ AI Model is OFFLINE - Using fallback mode")

print("=" * 60)


def analyze_image(image_path):
    """Analyze image using the model"""
    result = model_predictor.predict_image(image_path)
    return result['isAI'], result['confidence']


def analyze_video(video_path):
    """Analyze video using the model"""
    result = model_predictor.predict_video(video_path)
    return result['isAI'], result['confidence']


# ============================================================================
# FRONTEND ROUTES
# ============================================================================

@app.route('/')
def landing_page():
    return send_from_directory(str(FRONTEND_DIR), 'homes.html')


@app.route('/app')
def app_page():
    return send_from_directory(str(FRONTEND_DIR), 'home.html')


@app.route('/login')
def login_page():
    return send_from_directory(str(FRONTEND_DIR), 'login.html')


@app.route('/register')
def register_page():
    return send_from_directory(str(FRONTEND_DIR), 'register.html')


@app.route('/dashboard')
@login_required
def dashboard_page():
    return send_from_directory(str(FRONTEND_DIR), 'dashboard.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(str(FRONTEND_DIR), path)


# ============================================================================
# AUTH ROUTES
# ============================================================================

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    if not email or not password:
        return jsonify({'success': False, 'error': 'Email and password required'})
    
    if len(password) < 6:
        return jsonify({'success': False, 'error': 'Password must be at least 6 characters'})
    
    existing = User.query.filter_by(email=email).first()
    if existing:
        return jsonify({'success': False, 'error': 'Email already registered'})
    
    user = User(
        email=email,
        password_hash=generate_password_hash(password)
    )
    db.session.add(user)
    db.session.commit()
    
    session['user_id'] = user.id
    session['user_email'] = user.email
    
    return jsonify({
        'success': True,
        'user': user.to_dict(),
        'message': f'Welcome! You have unlimited analyses.'
    })


@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({'success': False, 'error': 'Invalid email or password'})
    
    session['user_id'] = user.id
    session['user_email'] = user.email
    
    return jsonify({
        'success': True,
        'user': user.to_dict()
    })


@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'success': True})


@app.route('/api/me', methods=['GET'])
def api_me():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        return jsonify({'success': False, 'error': 'User not found'}), 401
    
    return jsonify({
        'success': True,
        'user': user.to_dict()
    })


@app.route('/api/check_trial', methods=['GET'])
def check_trial():
    if 'user_id' in session:
        return jsonify({
            'is_logged_in': True,
            'free_remaining': 'unlimited'
        })
    
    # Anonymous user - check session counter
    trials_used = session.get('anonymous_trials_used', 0)
    remaining = max(0, FREE_TRIAL_LIMIT - trials_used)
    
    return jsonify({
        'is_logged_in': False,
        'free_remaining': remaining,
        'requires_login': remaining <= 0
    })


# ============================================================================
# ANALYSIS ROUTES
# ============================================================================

@app.route('/api/analyze', methods=['POST'])
@app.route('/api/analyze', methods=['POST'])
def analyze_media():
    # Check if user is logged in
    is_logged_in = 'user_id' in session
    user = None
    
    if is_logged_in:
        user = User.query.get(session['user_id'])
        if not user:
            session.clear()
            is_logged_in = False
    
    # For anonymous users, check free trial limit
    if not is_logged_in:
        if 'anonymous_trials_used' not in session:
            session['anonymous_trials_used'] = 0
        
        if session['anonymous_trials_used'] >= FREE_TRIAL_LIMIT:
            return jsonify({
                'success': False,
                'error': 'Free trial limit reached. Please create an account for unlimited access.',
                'limit_reached': True,
                'requires_login': True
            }), 403
    
    try:
        media_type = request.form.get('type', '').lower()
        
        if media_type == 'text':
            text_content = request.form.get('text', '')
            if not text_content.strip():
                return jsonify({'success': False, 'error': 'No text provided'})
            
            file_id = str(uuid.uuid4())
            filename = f"text_{file_id[:8]}.txt"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            is_ai, confidence = analyze_text_simple(text_content)
            explanation = generate_gemini_explanation('text', is_ai, confidence, text_content)
            
        elif media_type == 'image':
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'No file uploaded'})
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'})
            
            is_valid, detected_type, ext = validate_file_extension(file.filename)
            if not is_valid or detected_type != 'image':
                return jsonify({'success': False, 'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_IMAGE_EXTS)}'})
            
            file_id = str(uuid.uuid4())
            filename = f"{file_id}{ext}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            is_ai, confidence = analyze_image(file_path)
            explanation = generate_gemini_explanation('image', is_ai, confidence)
            
        elif media_type == 'video':
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'No file uploaded'})
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'})
            
            is_valid, detected_type, ext = validate_file_extension(file.filename)
            if not is_valid or detected_type != 'video':
                return jsonify({'success': False, 'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_VIDEO_EXTS)}'})
            
            file_id = str(uuid.uuid4())
            filename = f"{file_id}{ext}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save directly to final destination
            file.save(file_path)
            
            # Check video duration
            try:
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = total_frames / fps if fps > 0 else 0
                    cap.release()
                    
                    if duration > MAX_VIDEO_DURATION_SECONDS:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        return jsonify({'success': False, 'error': f'Video exceeds {MAX_VIDEO_DURATION_SECONDS} seconds. Current duration: {duration:.1f}s'})
            except Exception as ve:
                print(f"Warning: Could not read video duration: {ve}")
                # Continue anyway
            
            is_ai, confidence = analyze_video(file_path)
            explanation = generate_gemini_explanation('video', is_ai, confidence)
            
        else:
            return jsonify({'success': False, 'error': 'Invalid media type. Use image, video, or text.'})
        
        # Save analysis to database
        analysis = MediaAnalysis(
            user_id=user.id if user else None,
            filename=filename,
            media_type=media_type,
            file_path=file_path,
            is_ai=is_ai,
            confidence=confidence,
            explanation=explanation
        )
        db.session.add(analysis)
        
        # Track usage for anonymous users
        if not is_logged_in:
            session['anonymous_trials_used'] = session.get('anonymous_trials_used', 0) + 1
            session.modified = True
        
        db.session.commit()
        
        # Calculate remaining trials for response
        remaining_trials = 0
        if not is_logged_in:
            remaining_trials = max(0, FREE_TRIAL_LIMIT - session['anonymous_trials_used'])
        
        return jsonify({
            'success': True,
            'isAI': is_ai,
            'confidence': float(confidence),
            'analysis_id': analysis.id,
            'is_logged_in': is_logged_in,
            'remaining_trials': remaining_trials
        })
        
    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/explain/<analysis_id>', methods=['GET'])
@login_required
def get_explanation(analysis_id):
    analysis = MediaAnalysis.query.filter_by(id=analysis_id, user_id=session['user_id']).first()
    if not analysis:
        return jsonify({'success': False, 'error': 'Analysis not found'}), 404
    
    if not analysis.explanation:
        if analysis.media_type == 'text' and analysis.file_path:
            try:
                with open(analysis.file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            except:
                text_content = None
        else:
            text_content = None
        
        analysis.explanation = generate_gemini_explanation(
            analysis.media_type, analysis.is_ai, analysis.confidence, text_content
        )
        db.session.commit()
    
    return jsonify({
        'success': True,
        'explanation': analysis.explanation,
        'is_ai': analysis.is_ai,
        'confidence': analysis.confidence
    })


@app.route('/api/generate_report/<analysis_id>', methods=['GET'])
@login_required
def generate_report(analysis_id):
    analysis = MediaAnalysis.query.filter_by(id=analysis_id, user_id=session['user_id']).first()
    if not analysis:
        return jsonify({'success': False, 'error': 'Analysis not found'}), 404
    
    pdf_buffer = generate_pdf_report(analysis)
    
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'realytics_report_{analysis_id[:8]}.pdf'
    )


@app.route('/api/history', methods=['GET'])
@login_required
def get_history():
    analyses = MediaAnalysis.query.filter_by(user_id=session['user_id'])\
        .order_by(MediaAnalysis.created_at.desc()).limit(50).all()
    return jsonify([a.to_dict() for a in analyses])


@app.route('/api/delete/<analysis_id>', methods=['DELETE'])
@login_required
def delete_analysis(analysis_id):
    analysis = MediaAnalysis.query.filter_by(id=analysis_id, user_id=session['user_id']).first()
    if not analysis:
        return jsonify({'success': False, 'error': 'Analysis not found'}), 404
    
    # Delete the file if it exists
    if os.path.exists(analysis.file_path):
        try:
            os.remove(analysis.file_path)
        except:
            pass
    
    db.session.delete(analysis)
    db.session.commit()
    
    return jsonify({'success': True})


@app.route('/api/feedback', methods=['POST'])
@login_required
def submit_feedback():
    data = request.get_json()
    analysis_id = data.get('analysis_id')
    feedback_type = data.get('feedback_type')
    corrected_label = data.get('corrected_label')
    
    analysis = MediaAnalysis.query.filter_by(id=analysis_id, user_id=session['user_id']).first()
    if not analysis:
        return jsonify({'success': False, 'error': 'Analysis not found'}), 404
    
    feedback = Feedback(
        analysis_id=analysis_id,
        user_id=session['user_id'],
        feedback_type=feedback_type,
        corrected_label=corrected_label
    )
    db.session.add(feedback)
    db.session.commit()
    
    return jsonify({'success': True})


@app.route('/api/health', methods=['GET'])
def health_check():
    model_status = model_predictor.get_status()
    
    return jsonify({
        'status': 'healthy' if model_status.get('status') == 'online' else 'degraded',
        'model': model_status,
        'demo_mode': DEMO_MODE,
        'free_trial_limit': FREE_TRIAL_LIMIT,
        'explanations_enabled': ENABLE_EXPLANATIONS and GEMINI_AVAILABLE
    })


# ============================================================================
# INITIALIZATION
# ============================================================================

with app.app_context():
    db.create_all()
    print("✅ Database initialized")

print("\n" + "=" * 60)
print(" REALYTICS Server Running")
print("=" * 60)
print(f" URL: http://localhost:5000")
print(f" Demo Mode: {'ON' if DEMO_MODE else 'OFF'}")
print(f" AI Model: {'ONLINE' if model_predictor.is_initialized and not model_predictor.use_fallback else 'OFFLINE'}")
print(f" Free Trial Limit: {FREE_TRIAL_LIMIT} analysis for anonymous users")
print(f" Explanations: {'Gemini AI' if ENABLE_EXPLANATIONS and GEMINI_AVAILABLE else 'Simple'}")
print("=" * 60)

if __name__ == '__main__':
    # For Render, use the PORT environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
