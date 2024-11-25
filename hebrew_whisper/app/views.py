import time
from flask import Blueprint, request, jsonify, send_file, current_app
from flask_socketio import SocketIO, emit
import os
import subprocess
import torch
from models.whisper_model import WhisperModel

main_blueprint = Blueprint('main', __name__)
socketio = SocketIO()

@main_blueprint.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        print(f"Request files: {request.files}")
        print(f"Request form: {request.form}")
        
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        language = request.form.get('language', 'he')
        if not language:
            return jsonify({"error": "No language provided"}), 400

        uploads_dir = current_app.config['UPLOAD_FOLDER']
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)

        if file.filename.endswith('.mp3'):
            output_wav_path = os.path.splitext(file_path)[0] + '.wav'
            subprocess.run([
                'ffmpeg', '-y', '-i', file_path, output_wav_path
            ], check=True)
            os.remove(file_path)  # Remove the original MP3 file
            file_path = output_wav_path  # Update path to the new WAV file

        whisper_model = WhisperModel(model_name="medium")
        transcription = whisper_model.transcribe(file_path, language)

        transcript_folder = current_app.config.get('TRANSCRIPT_FOLDER')
        os.makedirs(transcript_folder, exist_ok=True)
        base_filename = os.path.splitext(file.filename)[0] + "_transcription"
        transcript_filepath = os.path.join(transcript_folder, f"{base_filename}.txt")

        with open(transcript_filepath, 'w', encoding='utf-8') as f:
            f.write(transcription)

        return send_file(transcript_filepath, as_attachment=True, download_name=os.path.basename(transcript_filepath))

    except Exception as e:
        current_app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


def convert_mp3_to_wav(input_path, output_path):
    subprocess.run([
        'ffmpeg', '-y', '-probesize', '50M', '-analyzeduration', '100M',
        '-i', input_path, output_path], check=True)


@main_blueprint.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200
