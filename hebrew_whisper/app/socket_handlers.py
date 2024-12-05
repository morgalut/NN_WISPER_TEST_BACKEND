from flask_socketio import SocketIO, emit
from models.whisper_model import WhisperModelSingleton
from utilities.file_handler import FileHandler
import torch


class SocketHandler:
    """Class to manage WebSocket event handlers."""

    def __init__(self, socketio: SocketIO, app):
        """
        Initialize the SocketHandler with the given SocketIO instance and Flask app.
        Args:
            socketio: The SocketIO instance.
            app: The Flask application instance.
        """
        self.socketio = socketio
        self.app = app
        self._register_events()

    def _register_events(self):
        """Register WebSocket event handlers."""
        @self.socketio.on('connect')
        def handle_connect():
            emit('log_message', {'message': 'Client connected successfully'})

        @self.socketio.on('transcribe')
        def handle_transcription(data):
            file_path = data.get('file_path')
            language = data.get('language', 'he')

            if not file_path:
                emit('error', {'error': 'File path is missing.'})
                return

            try:
                whisper_model = WhisperModelSingleton.get_instance()
                transcription = whisper_model.transcribe(file_path, language)

                # Emit progress updates (mocked here for demonstration)
                for i in range(1, 101, 10):
                    self.socketio.sleep(1)
                    emit('update_progress', {'progress': i, 'time_left': f'{100 - i} seconds'})

                # Emit transcription completion
                emit('transcription_complete', {'transcription': transcription})
            except Exception as e:
                emit('error', {'error': str(e)})

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

            emit('log_message', {'message': f"Processing {len(audio_files)} new audio files."})
            whisper_model = WhisperModelSingleton.get_instance(model_name="medium")  # Load model on-demand

            for audio_file in audio_files:
                audio_path = uploads_dir / audio_file
                try:
                    transcription = whisper_model.transcribe(audio_path, "he")
                    emit('log_message', {'message': f"Transcription complete for file: {audio_file}"})
                    FileHandler.delete_file(audio_path)
                except Exception as e:
                    emit('error', {'error': f"Error processing {audio_file}: {e}"})
            whisper_model.clean_up()  # Clear memory
            torch.cuda.empty_cache()  # Free GPU memory
