from flask_socketio import emit
from models.whisper_model import WhisperModelSingleton
from utilities.file_handler import FileHandler


class SocketHandler:
    """Class to manage WebSocket event handlers."""

    def __init__(self, socketio, app):
        """
        Initialize the SocketHandler with the SocketIO instance and Flask app.

        Args:
            socketio: Flask-SocketIO instance.
            app: Flask application instance.
        """
        self.socketio = socketio
        self.app = app
        self._register_events()

    def _register_events(self):
        """Register WebSocket events."""
        self.socketio.on_event('connect', self.handle_connect)
        self.socketio.on_event('start_transcription', self.handle_transcription)
        self.socketio.on_event('start_background_task', self.handle_background_task)

    def handle_connect(self):
        """Handle client connection."""
        emit('log_message', {'message': 'Client Connected'})

    def handle_transcription(self, data):
        """Handle transcription requests from clients."""
        file_path = data.get('file_path')
        language = data.get('language', 'he')

        if not file_path:
            emit('error', {'error': 'File path is missing.'})
            return

        whisper_model = WhisperModelSingleton.get_instance(model_name="medium")
        try:
            emit('log_message', {'message': f"Starting transcription of {file_path}"})
            transcription = whisper_model.transcribe(file_path, language)
            emit('transcription_complete', {
                'message': 'File has been successfully transcribed.',
                'transcription': transcription,
            })
        except Exception as e:
            emit('error', {'error': str(e)})
        finally:
            whisper_model.clean_up()

    def handle_background_task(self):
        """Start the background task via WebSocket."""
        self.socketio.start_background_task(self._background_task, self.app.app_context())

    def _background_task(self, app_context):
        """Background task to process new audio files."""
        with app_context:
            uploads_dir = self.app.config.get('UPLOAD_FOLDER', './uploads')
            audio_files = FileHandler.check_new_files(uploads_dir, ('.wav',))

            if not audio_files:
                return

            self.socketio.emit('log_message', {'message': f"Processing {len(audio_files)} new audio files."})
            whisper_model = WhisperModelSingleton.get_instance(model_name="medium")

            for audio_file in audio_files:
                audio_path = uploads_dir / audio_file
                try:
                    transcription = whisper_model.transcribe(audio_path, "he")
                    self.socketio.emit('log_message', {'message': f"Transcription complete for file: {audio_file}"})
                    FileHandler.delete_file(audio_path)
                except Exception as e:
                    self.socketio.emit('error', {'error': f"Error processing {audio_file}: {e}"})
