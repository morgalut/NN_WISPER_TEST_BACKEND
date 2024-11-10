import os

class Config:
    DEBUG = True
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Set to root folder "hebrew_whisper"
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data/uploads')
    TRANSCRIPT_FOLDER = os.path.join(BASE_DIR, 'data/transcripts')
    FEEDBACK_FOLDER = os.path.join(BASE_DIR, 'data/feedback')
    TRAINING_DATA_FOLDER = os.path.join(BASE_DIR, 'data/training_data')
    PERFECT_TRAINING_FOLDER = os.path.join(BASE_DIR, 'data/training_data/perfect_training')
