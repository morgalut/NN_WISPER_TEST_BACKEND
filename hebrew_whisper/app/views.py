import asyncio
from flask import Blueprint, Response, json, request, jsonify, send_file, current_app,make_response
from utilities.text_normalizer import AudioPreprocessor, TextNormalizer
from models.whisper_model import WhisperModel
from services.transcription_service import TranscriptionService
from services.training_service import TrainingService
from utilities.file_handler import FileHandler
import os

main_blueprint = Blueprint('main', __name__)


@main_blueprint.route('/transcribe', methods=['POST'])
async def transcribe_audio():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    preprocessor = AudioPreprocessor('base', current_app.config['TRAINING_DATA_FOLDER'])
    data, rate = preprocessor.preprocess_audio(file_path)

    try:
        whisper_model = WhisperModel()
        transcript = await whisper_model.transcribe(file_path, language="he")
        
        # Define the initial path for saving the transcription file
        transcript_folder = current_app.config.get('TRANSCRIPT_FOLDER', './transcripts')
        base_filename = f"{os.path.splitext(file.filename)[0]}_transcription"
        
        # Ensure the transcript folder exists
        os.makedirs(transcript_folder, exist_ok=True)
        
        # Check for existing file with the same name and add a number if necessary
        counter = 1
        transcript_filepath = os.path.join(transcript_folder, f"{base_filename}.txt")
        while os.path.exists(transcript_filepath):
            transcript_filepath = os.path.join(transcript_folder, f"{base_filename}_{counter}.txt")
            counter += 1

        # Write the transcription to a text file with Hebrew content
        with open(transcript_filepath, 'w', encoding='utf-8') as f:
            f.write(transcript)
    finally:
        os.remove(file_path)  # Clean up uploaded file

    # Return the text file as a downloadable response
    transcript_filename = os.path.basename(transcript_filepath)
    return send_file(transcript_filepath, as_attachment=True, download_name=transcript_filename)


@main_blueprint.route('/feedback', methods=['POST'])
def receive_feedback():
    return TrainingService.receive_feedback(request)
