from flask import Blueprint, request, jsonify, send_file, current_app
import os
import subprocess
import torch
from models.whisper_model import WhisperModel, WhisperModelSingleton

# Define the Blueprint with a URL prefix
main_blueprint = Blueprint('main', __name__, url_prefix='/api')  # Ensure routes are prefixed with `/api`


@main_blueprint.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Handle audio file upload and transcription."""
    file_path = None  # Initialize file_path for cleanup purposes
    try:
        # Validate file upload
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Retrieve the language or use the default
        language = request.form.get('language', 'he')

        # Define the uploads directory
        uploads_dir = current_app.config.get('UPLOAD_FOLDER', './uploads')
        os.makedirs(uploads_dir, exist_ok=True)

        # Save the uploaded file
        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)

        # Convert MP3 to WAV if needed
        file_path = convert_to_wav_if_needed(file_path, file.filename)

        # Explicitly create a new WhisperModel instance
        current_app.logger.info("Creating a new Whisper model instance for transcription...")
        whisper_model = WhisperModel(model_name="medium", beam_size=5, temperature=0.5)

        # Perform transcription
        transcription = whisper_model.transcribe(file_path, language)

        # Save transcription to a file and return it to the client
        return save_and_send_transcription(file.filename, transcription)
    except subprocess.CalledProcessError as e:
        current_app.logger.error(f"Error converting file with FFmpeg: {str(e)}", exc_info=True)
        return jsonify({"error": "Error converting file. Ensure FFmpeg is installed."}), 500
    except Exception as e:
        current_app.logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        # Ensure uploaded file is removed after processing
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        torch.cuda.empty_cache()  # Clear GPU memory


def convert_to_wav_if_needed(file_path, filename):
    """Convert MP3 file to WAV if necessary."""
    if filename.endswith('.mp3'):
        output_wav_path = os.path.splitext(file_path)[0] + '.wav'
        subprocess.run(['ffmpeg', '-y', '-i', file_path, output_wav_path], check=True)
        os.remove(file_path)  # Remove the original MP3 file
        return output_wav_path
    return file_path


def save_and_send_transcription(filename, transcription):
    """Save transcription to a file and send it as a response."""
    # Define the transcripts directory
    transcript_folder = current_app.config.get('TRANSCRIPT_FOLDER', './transcripts')
    os.makedirs(transcript_folder, exist_ok=True)

    # Save transcription
    base_filename = os.path.splitext(filename)[0]
    transcript_filepath = os.path.join(transcript_folder, f"{base_filename}_transcription.txt")
    with open(transcript_filepath, 'w', encoding='utf-8') as f:
        f.write(transcription)

    # Send the transcription file to the client
    return send_file(transcript_filepath, as_attachment=True, download_name=os.path.basename(transcript_filepath))
