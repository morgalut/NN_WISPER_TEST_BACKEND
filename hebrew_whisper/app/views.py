from flask import Blueprint, request, jsonify, send_file, current_app
import os
import subprocess
import torch
from models.whisper_model import WhisperModel, WhisperModelSingleton

main_blueprint = Blueprint('main', __name__)

@main_blueprint.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Handle file upload and transcription."""
    file_path = None
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        language = request.form.get('language', 'he')
        uploads_dir = current_app.config.get('UPLOAD_FOLDER', './uploads')
        os.makedirs(uploads_dir, exist_ok=True)

        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)

        # Convert MP3 to WAV if necessary
        if file.filename.endswith('.mp3'):
            output_wav_path = os.path.splitext(file_path)[0] + '.wav'
            subprocess.run(['ffmpeg', '-y', '-i', file_path, output_wav_path], check=True)
            os.remove(file_path)
            file_path = output_wav_path

        whisper_model = WhisperModelSingleton.get_instance()
        transcription = whisper_model.transcribe(file_path, language)

        # Save transcription to file
        transcript_folder = current_app.config.get('TRANSCRIPT_FOLDER', './transcripts')
        os.makedirs(transcript_folder, exist_ok=True)
        base_filename = os.path.splitext(file.filename)[0]
        transcript_filepath = os.path.join(transcript_folder, f"{base_filename}_transcription.txt")

        with open(transcript_filepath, 'w', encoding='utf-8') as f:
            f.write(transcription)

        return send_file(transcript_filepath, as_attachment=True, download_name=os.path.basename(transcript_filepath))
    except subprocess.CalledProcessError as e:
        current_app.logger.error(f"Error converting file with FFmpeg: {str(e)}", exc_info=True)
        return jsonify({"error": "Error converting file. Ensure FFmpeg is installed."}), 500
    except Exception as e:
        current_app.logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        torch.cuda.empty_cache()

@main_blueprint.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200
