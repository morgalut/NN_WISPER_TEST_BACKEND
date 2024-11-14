from flask import Blueprint, request, jsonify, send_file, current_app
import os
import numpy as np
import torch
from models.whisper_model import WhisperModel
from utilities.text_normalizer import AudioPreprocessor
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

main_blueprint = Blueprint('main', __name__)

@main_blueprint.route('/transcribe', methods=['POST'])
def transcribe_audio():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Save uploaded file
    uploads_dir = current_app.config['UPLOAD_FOLDER']
    os.makedirs(uploads_dir, exist_ok=True)
    file_path = os.path.join(uploads_dir, file.filename)
    file.save(file_path)

    # Convert MP3 to WAV if necessary
    if file.filename.endswith('.mp3'):
        try:
            audio = AudioSegment.from_mp3(file_path)
            wav_path = os.path.splitext(file_path)[0] + '.wav'
            audio.export(wav_path, format='wav')
            os.remove(file_path)  # Remove the original MP3 file
            file_path = wav_path  # Update path to the new WAV file
        except CouldntDecodeError:
            os.remove(file_path)
            return jsonify({"error": "Could not decode MP3 file. Please check file integrity or try another file format."}), 400

    try:
        # Initialize Whisper model
        whisper_model = WhisperModel(model_name="large-v2")
        
        # Transcribe using the file path, not the tensor directly
        transcription = whisper_model.transcribe(file_path, language="he")

        # Save transcription
        transcript_folder = current_app.config.get('TRANSCRIPT_FOLDER')
        os.makedirs(transcript_folder, exist_ok=True)
        base_filename = os.path.splitext(file.filename)[0] + "_transcription"
        transcript_filepath = os.path.join(transcript_folder, f"{base_filename}.txt")

        # Ensure unique file name if necessary
        counter = 1
        while os.path.exists(transcript_filepath):
            transcript_filepath = os.path.join(transcript_folder, f"{base_filename}_{counter}.txt")
            counter += 1

        with open(transcript_filepath, 'w', encoding='utf-8') as f:
            f.write(transcription)

    finally:
        # Clean up
        os.remove(file_path)

    # Return transcription file
    return send_file(transcript_filepath, as_attachment=True, download_name=os.path.basename(transcript_filepath))
