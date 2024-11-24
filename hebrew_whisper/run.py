import os
from flask import Flask
from threading import Thread
from flask_cors import CORS  # Import CORS
from app.views import main_blueprint  # Correct the import path for the blueprint
from models.whisper_model import WhisperModel
from utilities.file_handler import FileHandler

def create_app():
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__)
    app.config.from_object('app.config.Config')
    
    # Enable CORS
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Register the directly imported blueprint
    app.register_blueprint(main_blueprint)
    
    return app

app = create_app()

def background_task(app):
    """Run background tasks that don't block the Flask application."""
    with app.app_context():
        uploads_dir = app.config['UPLOAD_FOLDER']
        new_audio_files = FileHandler.check_new_files(uploads_dir, ('.wav',))
        print("New audio files:", new_audio_files)
        
        # Initialize Whisper model
        whisper_model = WhisperModel(model_name="medium")  # Use medium model for optimized memory usage

        # Process files in batches of 2
        for i in range(0, len(new_audio_files), 2):  # Process 2 files at a time
            batch = new_audio_files[i:i + 2]
            for audio_file in batch:
                try:
                    audio_path = os.path.join(uploads_dir, audio_file)
                    print(f"Transcribing {audio_file} using WhisperModel...")
                    transcription = whisper_model.transcribe(audio_path, language="he")
                    print(f"Transcription for {audio_file}: {transcription}")
                    FileHandler.delete_file(audio_path)  # Cleanup after processing
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")

        # Clean up Whisper model to free GPU memory
        whisper_model.clean_up()

def setup_background_tasks(app):
    """Set up and run background tasks in a separate thread."""
    thread = Thread(target=background_task, args=(app,), daemon=True)
    thread.start()

if __name__ == "__main__":
    # Use the Render-assigned PORT or default to 5000 for local testing
    port = int(os.getenv('PORT', 10000))  # Default is 10000, as Render suggests
    app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])
    print(f"Server running on http://0.0.0.0:{port}")
