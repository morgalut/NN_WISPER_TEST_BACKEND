# Apply monkey patching before importing anything else
import eventlet
eventlet.monkey_patch()

import os
from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from app.views import main_blueprint
from models.whisper_model import WhisperModelSingleton
from utilities.file_handler import FileHandler


# Create and configure Flask application
def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')
    
    # Configure SocketIO with eventlet for asynchronous operations
    socketio = SocketIO(app, cors_allowed_origins="*", logger=True, async_mode='eventlet')
    CORS(app)
    app.register_blueprint(main_blueprint)
    return app, socketio


app, socketio = create_app()


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('log_message', {'message': 'Client Connected'})


@socketio.on('start_transcription')
def handle_transcription(data):
    """Handle transcription request from client."""
    file_path = data.get('file_path')
    language = data.get('language', 'he')

    if not file_path:
        emit('error', {'error': 'File path is missing.'})
        return

    whisper_model = WhisperModelSingleton.get_instance(model_name="medium")
    try:
        emit('log_message', {'message': f"Starting transcription of {os.path.basename(file_path)}"})
        transcription = whisper_model.transcribe(file_path, language)
        emit('transcription_complete', {
            'message': 'File has been successfully transcribed.',
            'transcription': transcription
        })
    except Exception as e:
        emit('error', {'error': str(e)})
    finally:
        whisper_model.clean_up()


def background_task(app_context):
    """Background task to process new audio files."""
    with app_context:  # Ensure the Flask application context is active
        uploads_dir = app.config.get('UPLOAD_FOLDER', './uploads')
        audio_files = FileHandler.check_new_files(uploads_dir, ('.wav',))

        if not audio_files:
            return

        socketio.emit('log_message', {'message': f"Processing {len(audio_files)} new audio files."})
        whisper_model = WhisperModelSingleton.get_instance(model_name="medium")

        for audio_file in audio_files:
            audio_path = os.path.join(uploads_dir, audio_file)
            try:
                transcription = whisper_model.transcribe(audio_path, "he")
                socketio.emit('log_message', {'message': f"Transcription complete for file: {audio_file}"})
                FileHandler.delete_file(audio_path)
            except Exception as e:
                socketio.emit('error', {'error': f"Error processing {audio_file}: {e}"})


@socketio.on('start_background_task')
def handle_background_task():
    """Start the background task via WebSocket."""
    socketio.start_background_task(background_task, app.app_context())


if __name__ == "__main__":
    try:
        port = int(os.getenv('PORT', 10000))
        socketio.run(app, host='0.0.0.0', port=port, log_output=True)
        print(f"Server running on http://0.0.0.0:{port}")
    except Exception as e:
        print(f"Error starting the server: {e}")
