import os
from flask import Flask
from threading import Thread
from app import create_app
from models.whisper_model import WhisperModel
from utilities.file_handler import FileHandler

app = create_app()

def background_task(app):
    """Run background tasks that don't block the Flask application."""
    with app.app_context():
        uploads_dir = app.config['UPLOAD_FOLDER']
        new_audio_files = FileHandler.check_new_files(uploads_dir, ('.wav',))
        print("New audio files:", new_audio_files)
        whisper_model = WhisperModel('base')  # Assuming this setup waits for model loading appropriately

        for audio_file in new_audio_files:
            audio_path = os.path.join(uploads_dir, audio_file)
            print(f"Transcribing {audio_file} using WhisperModel directly...")
            # Assuming the transcribe method can handle synchronous calls or has been adapted accordingly
            transcription = whisper_model.get_model().transcribe(audio_path)
            print(f"Transcription for {audio_file}: {transcription}")

def setup_background_tasks(app):
    """Set up and run background tasks in a separate thread."""
    thread = Thread(target=background_task, args=(app,))
    thread.start()

if __name__ == "__main__":
    app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'True') == 'True'
    setup_background_tasks(app)
    app.run(host='0.0.0.0', port=5000,debug=app.config['DEBUG'])
