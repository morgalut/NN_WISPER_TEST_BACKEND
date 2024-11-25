import time
from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os
from threading import Thread
from app.views import main_blueprint
from models.whisper_model import WhisperModel
from utilities.file_handler import FileHandler

def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')
    
    # Initialize SocketIO with explicit configurations
    socketio = SocketIO(app, cors_allowed_origins="*", logger=True, manage_session=False)

    CORS(app, resources={r"/*": {"origins": "*"}})
    app.register_blueprint(main_blueprint)

    return app, socketio

app, socketio = create_app()

@socketio.on('connect')
def handle_connect():
    emit('log_message', {'message': 'Client Connected'})

@socketio.on('start_transcription')
def handle_transcription(data):
    file_path = data['file_path']
    language = data.get('language', 'he')
    whisper_model = WhisperModel(model_name="medium")
    try:
        emit('log_message', {'message': f"Starting transcription of {os.path.basename(file_path)}"})
        total_steps = 10
        for step in range(total_steps):
            time.sleep(1)
            progress = ((step + 1) / total_steps) * 100
            time_left = (total_steps - (step + 1)) * 1
            emit('update_progress', {'progress': progress, 'time_left': f"{time_left} seconds remaining"})

        transcription = whisper_model.transcribe(file_path, language)
        emit('log_message', {'message': "Transcription completed successfully."})
        emit('transcription_complete', {'message': 'File has been successfully transcribed', 'transcription': transcription})
    except Exception as e:
        emit('error', {'error': str(e)})
    finally:
        whisper_model.clean_up()

def setup_background_tasks():
    thread = Thread(target=background_task, daemon=True)
    thread.start()

def background_task():
    uploads_dir = app.config['UPLOAD_FOLDER']
    new_audio_files = FileHandler.check_new_files(uploads_dir, ('.wav',))
    socketio.emit('log_message', {'message': f"Found {len(new_audio_files)} new audio files."})
    whisper_model = WhisperModel(model_name="medium")

    for audio_file in new_audio_files:
        audio_path = os.path.join(uploads_dir, audio_file)
        try:
            socketio.emit('log_message', {'message': f"Processing file: {audio_file}"})
            transcription = whisper_model.transcribe(audio_path, "he")
            socketio.emit('log_message', {'message': f"Completed transcription for file: {audio_file}"})
            FileHandler.delete_file(audio_path)
        except Exception as e:
            socketio.emit('error', {'error': f"Error processing {audio_file}: {e}"})

        whisper_model.clean_up()

if __name__ == "__main__":
    try:
        setup_background_tasks()
        port = int(os.getenv('PORT', 10000))
        socketio.run(app, host='0.0.0.0', port=port, use_reloader=False, log_output=True)
        print(f"Server running on http://0.0.0.0:{port}")
    except Exception as e:
        print(f"Error starting the server: {e}")
