from flask import Blueprint, request, jsonify, send_file, current_app
import os
import torch
from models.whisper_model import WhisperModel

# Define the Blueprint with a URL prefix
main_blueprint = Blueprint('main', __name__, url_prefix='/api')


class TranscriptionService:
    """Service class to handle audio file transcription and related tasks."""

    def __init__(self, uploads_dir='./uploads', transcript_dir='./transcripts', model_name='medium'):
        self.uploads_dir = uploads_dir
        self.transcript_dir = transcript_dir
        self.model_name = model_name

        # Ensure directories exist
        os.makedirs(self.uploads_dir, exist_ok=True)
        os.makedirs(self.transcript_dir, exist_ok=True)

        # Initialize the Whisper model
        self.whisper_model = WhisperModel(model_name=self.model_name)

    def save_uploaded_file(self, file):
        """Save the uploaded file to the uploads directory."""
        file_path = os.path.join(self.uploads_dir, file.filename)
        try:
            file.save(file_path)
            return file_path
        except Exception as e:
            raise RuntimeError(f"Failed to save file: {e}")

    def transcribe(self, file_path, language):
        """Transcribe the audio file using the Whisper model."""
        try:
            return self.whisper_model.transcribe(file_path, language)
        except Exception as e:
            raise RuntimeError(f"Error during transcription: {e}")

    def save_transcription(self, filename, transcription):
        """Save the transcription to a text file."""
        base_filename = os.path.splitext(filename)[0]
        transcript_filepath = os.path.join(self.transcript_dir, f"{base_filename}_transcription.txt")
        with open(transcript_filepath, 'w', encoding='utf-8') as f:
            f.write(transcription)
        return transcript_filepath


@main_blueprint.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """API endpoint to handle audio file transcription."""
    file_path = None
    try:
        # Validate the uploaded file
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Retrieve language from the request or use the default
        language = request.form.get('language', 'he')

        # Initialize the transcription service
        transcription_service = TranscriptionService(
            uploads_dir=current_app.config.get('UPLOAD_FOLDER', './uploads'),
            transcript_dir=current_app.config.get('TRANSCRIPT_FOLDER', './transcripts')
        )

        # Save the uploaded file
        file_path = transcription_service.save_uploaded_file(file)

        # Transcribe the audio file
        transcription = transcription_service.transcribe(file_path, language)

        # Save and return the transcription
        transcript_filepath = transcription_service.save_transcription(file.filename, transcription)
        return send_file(transcript_filepath, as_attachment=True, download_name=os.path.basename(transcript_filepath))

    except Exception as e:
        current_app.logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        # Ensure the uploaded file is removed after processing
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        torch.cuda.empty_cache()  # Clear GPU memory


@main_blueprint.route('/health', methods=['GET'])
def health_check():
    """Endpoint to verify service and dependency health."""
    return jsonify({
        "status": "OK"
    }), 200
