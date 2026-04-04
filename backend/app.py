from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
import uuid
from datetime import datetime
import cv2
import numpy as np  # ✅ ADDED - was missing!
from PIL import Image
import json  # ✅ ADDED - was missing!
from pathlib import Path
from model_predictor import get_model_predictor, initialize_model as init_predictor  # ✅ ADDED - use external predictor

app = Flask(__name__)
CORS(app)

# Configuration - add DEMO_MODE flag
DEMO_MODE = os.environ.get('DEMO_MODE', 'false').lower() == 'true'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///analysis.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

db = SQLAlchemy(app)

# Database Model
class MediaAnalysis(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    media_type = db.Column(db.String(10), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    is_ai = db.Column(db.Boolean, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'media_type': self.media_type,
            'is_ai': self.is_ai,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat()
        }

# Initialize database
with app.app_context():
    try:
        db.create_all()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")

# ✅ REMOVED the entire AIModelPredictor class (lines 69-417 from original)
# ✅ Now using model_predictor.py instead

# ✅ UPDATED analysis functions to use imported predictor
def analyze_image(image_path):
    """Analyze image for AI vs Human content"""
    try:
        predictor = get_model_predictor()
        result = predictor.predict_image(image_path)
        return result['isAI'], result['confidence']
    except Exception as e:
        print(f"❌ Image analysis error: {e}")
        return False, 50.0

def analyze_video(video_path):
    """Analyze video for AI vs Human content"""
    try:
        predictor = get_model_predictor()
        result = predictor.predict_video(video_path)
        return result['isAI'], result['confidence']
    except Exception as e:
        print(f"❌ Video analysis error: {e}")
        return False, 50.0

def analyze_text(text_content):
    """Analyze text for AI vs Human content"""
    try:
        predictor = get_model_predictor()
        result = predictor.predict_text(text_content)
        return result['isAI'], result['confidence']
    except Exception as e:
        print(f"❌ Text analysis error: {e}")
        return False, 50.0

@app.route('/')
def serve_frontend():
    return send_from_directory('../frontend', 'home.html')

@app.route('/<path:path>')
def serve_static_files(path):
    """Serve static files from frontend directory"""
    return send_from_directory('../frontend', path)

@app.route('/analyze', methods=['POST'])
def analyze_media():
    try:
        if 'file' not in request.files and 'text' not in request.form:
            return jsonify({'success': False, 'error': 'No file or text provided'})
        
        media_type = request.form.get('type', 'image')
        
        if media_type == 'text':
            text_content = request.form.get('text', '')
            if not text_content.strip():
                return jsonify({'success': False, 'error': 'No text provided'})
            
            file_id = str(uuid.uuid4())
            filename = f"text_analysis_{file_id[:8]}.txt"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            is_ai, confidence = analyze_text(text_content)
            
        else:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'})
            
            file_id = str(uuid.uuid4())
            file_extension = os.path.splitext(file.filename)[1]
            filename = f"{file_id}{file_extension}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(file_path)
            
            if media_type == 'image':
                is_ai, confidence = analyze_image(file_path)
            else:  # video
                is_ai, confidence = analyze_video(file_path)
        
        # Save to database
        try:
            analysis = MediaAnalysis(
                id=file_id,
                filename=filename,
                media_type=media_type,
                file_path=file_path,
                is_ai=is_ai,
                confidence=confidence
            )
            db.session.add(analysis)
            db.session.commit()
        except Exception as db_error:
            print(f"Database error: {db_error}")
        
        return jsonify({
            'success': True,
            'isAI': is_ai,
            'confidence': float(confidence),
            'file_id': file_id
        })
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/history', methods=['GET'])
def get_history():
    try:
        analyses = MediaAnalysis.query.order_by(MediaAnalysis.created_at.desc()).limit(20).all()
        return jsonify([analysis.to_dict() for analysis in analyses])
    except Exception as e:
        print(f"History error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health', methods=['GET'])
def health_check():
    predictor = get_model_predictor()
    metrics = predictor.get_metrics()
    return jsonify({
        'status': 'healthy',
        'database': 'connected' if db else 'disconnected',
        'upload_folder': os.path.exists(app.config['UPLOAD_FOLDER']),
        'model_status': 'loaded' if not metrics['model_loaded'] else 'fallback',
        'demo_mode': DEMO_MODE,
        'metrics': metrics
    })

@app.route('/set_demo_mode/<int:mode>', methods=['POST'])
def set_demo_mode(mode):
    global DEMO_MODE
    DEMO_MODE = bool(mode)
    return jsonify({
        'success': True,
        'demo_mode': DEMO_MODE,
        'message': f'Demo mode {"enabled" if DEMO_MODE else "disabled"}'
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    predictor = get_model_predictor()
    return jsonify(predictor.get_metrics())

# ✅ REMOVED the old initialize_model() function
# ✅ REMOVED the global model_predictor variable

# Initialize model when the app starts
print("🚀 Starting AI vs Human Media & Text Detector Server...")
print("📁 Upload folder:", app.config['UPLOAD_FOLDER'])
print("🗄️  Database:", app.config['SQLALCHEMY_DATABASE_URI'])
print("🎮 Demo Mode:", "ENABLED" if DEMO_MODE else "DISABLED")
print("🔧 Initializing AI Model...")

if init_predictor():  # ✅ Using imported function
    print("✅ AI Model initialized successfully!")
else:
    print("⚠️ AI Model initialized in fallback mode.")

print("🌐 Server running at: http://localhost:5000")
print("📝 Text analysis: Enabled")
print("🖼️  Image analysis: Enabled")
print("🎥 Video analysis: Enabled")

if __name__ == '__main__':
    app.run(debug=True, port=5000)