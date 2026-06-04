"""
Database Models for AI Detector
"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid

db = SQLAlchemy()


class User(db.Model):
    """User account"""
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Track free trial usage
    free_analyses_used = db.Column(db.Integer, default=0)
    
    # Relationships
    analyses = db.relationship('MediaAnalysis', backref='user', lazy=True)
    feedbacks = db.relationship('Feedback', backref='user', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'free_analyses_remaining': max(0, 1 - self.free_analyses_used),
            'created_at': self.created_at.isoformat()
        }


class MediaAnalysis(db.Model):
    """Store analysis results"""
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=True)
    filename = db.Column(db.String(255), nullable=False)
    media_type = db.Column(db.String(10), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    is_ai = db.Column(db.Boolean, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    explanation = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'media_type': self.media_type,
            'is_ai': self.is_ai,
            'confidence': self.confidence,
            'explanation': self.explanation,
            'created_at': self.created_at.isoformat()
        }


class Feedback(db.Model):
    """Store user feedback for retraining"""
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id = db.Column(db.String(36), db.ForeignKey('media_analysis.id'), nullable=False)
    user_id = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=True)
    feedback_type = db.Column(db.String(10), nullable=False)
    corrected_label = db.Column(db.Boolean, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    analysis = db.relationship('MediaAnalysis', backref='feedbacks')