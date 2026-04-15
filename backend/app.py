import os
import sys
import uuid
import random
import subprocess
from datetime import datetime
from pathlib import Path

import magic
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

sys.path.append(str(Path(__file__).parent))

from config import (
    MODELS_DIR, UPLOAD_FOLDER, DATABASE_URI,
    MAX_CONTENT_LENGTH, ALLOWED_IMAGE_MIMES, ALLOWED_IMAGE_EXTS,
    DEMO_MODE, CLAMAV_ENABLED, CLAMAV_SOCKET, RETRAIN_THRESHOLD
)
from model_prediction import EnsemblePredictor

# ----------------------------------------------------------------------
# Flask app setup
# ----------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ----------------------------------------------------------------------
# Database Models
# ----------------------------------------------------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    tests = db.relationship('TestHistory', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class TestHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_name = db.Column(db.String(200))
    prediction = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    correct_guess = db.Column(db.Boolean, default=None)

class Feedback(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    analysis_id = db.Column(db.String(36), nullable=False)
    feedback_type = db.Column(db.String(10), nullable=False)
    corrected_label = db.Column(db.Boolean, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

# ----------------------------------------------------------------------
# File Validation
# ----------------------------------------------------------------------
def validate_file_mime(file_stream):
    try:
        mime = magic.from_buffer(file_stream.read(1024), mime=True)
        file_stream.seek(0)
        return mime, mime in ALLOWED_IMAGE_MIMES
    except Exception as e:
        print(f"MIME error: {e}")
        return None, False

def validate_file_clamav(file_stream):
    if not CLAMAV_ENABLED:
        return True, "ClamAV disabled"
    try:
        import pyclamd
        cd = pyclamd.ClamdUnixSocket(CLAMAV_SOCKET)
        file_stream.seek(0)
        result = cd.scan_stream(file_stream.read())
        file_stream.seek(0)
        if result:
            return False, f"Virus detected"
        return True, "Clean"
    except:
        return True, "ClamAV unavailable"

def validate_file(filename, file_stream):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        return False, f"Invalid extension. Use: {', '.join(ALLOWED_IMAGE_EXTS)}"
    
    mime, is_allowed = validate_file_mime(file_stream)
    if not is_allowed:
        return False, f"Invalid file type: {mime}"
    
    if CLAMAV_ENABLED:
        is_clean, msg = validate_file_clamav(file_stream)
        if not is_clean:
            return False, msg
    
    return True, "Valid file"

# ----------------------------------------------------------------------
# Model Predictor
# ----------------------------------------------------------------------
model_predictor = None

def initialize_model():
    global model_predictor
    try:
        model_predictor = EnsemblePredictor(models_dir=MODELS_DIR, demo_mode=DEMO_MODE)
        return model_predictor.is_initialized
    except Exception as e:
        print(f"Model init failed: {e}")
        model_predictor = EnsemblePredictor(models_dir=MODELS_DIR, demo_mode=DEMO_MODE)
        return False

# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error="Username exists")
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for('dashboard'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=current_user.username)

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Validate file
    is_valid, msg = validate_file(file.filename, file)
    if not is_valid:
        return jsonify({'error': msg}), 400
    
    user_guess = request.form.get('user_guess', None)
    temp_path = UPLOAD_FOLDER / f"temp_{current_user.id}_{datetime.now().timestamp()}.jpg"
    file.save(str(temp_path))
    
    try:
        if model_predictor is None:
            initialize_model()
        
        result = model_predictor.predict_image(str(temp_path))
        if not result:
            return jsonify({'error': 'Prediction failed'}), 500
        
        prediction = result['predicted_class']
        confidence = result['confidence'] / 100.0  # Convert to 0-1
        
        correct_guess = None
        if user_guess:
            correct_guess = (user_guess.lower() == prediction.lower())
        
        test_record = TestHistory(
            user_id=current_user.id,
            image_name=file.filename,
            prediction=prediction,
            confidence=confidence,
            correct_guess=correct_guess
        )
        db.session.add(test_record)
        db.session.commit()
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence * 100,
            'correct_guess': correct_guess
        })
    finally:
        if temp_path.exists():
            temp_path.unlink()

@app.route('/api/history')
@login_required
def api_history():
    tests = TestHistory.query.filter_by(user_id=current_user.id)\
        .order_by(TestHistory.timestamp.desc()).limit(50).all()
    
    history = []
    for test in tests:
        history.append({
            'id': test.id,
            'image_name': test.image_name,
            'prediction': test.prediction,
            'confidence': round(test.confidence * 100, 2),
            'correct_guess': test.correct_guess,
            'timestamp': test.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    total = len(history)
    correct = sum(1 for t in history if t['correct_guess'] is True)
    
    stats = {
        'total_tests': total,
        'correct_guesses': correct,
        'incorrect_guesses': total - correct,
        'accuracy': round((correct / total * 100) if total > 0 else 0, 2)
    }
    
    return jsonify({'history': history, 'stats': stats})

@app.route('/api/clear_history', methods=['POST'])
@login_required
def clear_history():
    TestHistory.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    return jsonify({'message': 'History cleared'})

@app.route('/api/feedback', methods=['POST'])
@login_required
def submit_feedback():
    data = request.get_json()
    feedback = Feedback(
        id=str(uuid.uuid4()),
        analysis_id=data.get('analysis_id'),
        feedback_type=data.get('feedback_type'),
        corrected_label=data.get('corrected_label')
    )
    db.session.add(feedback)
    db.session.commit()
    
    # Trigger retraining if threshold reached
    if Feedback.query.count() >= RETRAIN_THRESHOLD:
        subprocess.Popen(["python", str(Path(__file__).parent / "retrain.py")])
    
    return jsonify({'success': True})

@app.route('/health', methods=['GET'])
def health_check():
    model_status = "loaded" if model_predictor and not model_predictor.use_fallback else "not_loaded"
    return jsonify({
        'status': 'healthy' if model_status == 'loaded' else 'degraded',
        'database': 'connected' if db else 'disconnected',
        'model_status': model_status,
        'demo_mode': DEMO_MODE,
        'fallback_active': model_predictor.use_fallback if model_predictor else False
    })

if __name__ == '__main__':
    initialize_model()
    app.run(debug=True, port=5000)