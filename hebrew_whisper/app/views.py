from flask import Blueprint, request, jsonify, send_file, current_app
import os
import subprocess
import torch
from models.whisper_model import WhisperModel
from utilities.text_normalizer import AudioPreprocessor
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

main_blueprint = Blueprint('main', __name__)

@main_blueprint.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        print(f"Request files: {request.files}")
        print(f"Request form: {request.form}")
        
        file = request.files.get('file')
        if not file:
            print("No file found in request.files")
            return jsonify({"error": "No file uploaded"}), 400

        language = request.form.get('language', 'he')
        if not language:
            print("No language provided in request.form")
            return jsonify({"error": "No language provided"}), 400

        # Save uploaded file
        uploads_dir = current_app.config['UPLOAD_FOLDER']
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = os.path.join(uploads_dir, file.filename)
        print(f"Saving file to: {file_path}")
        file.save(file_path)

        # Convert MP3 to WAV if necessary using FFmpeg directly
        if file.filename.endswith('.mp3'):
            try:
                output_wav_path = os.path.splitext(file_path)[0] + '.wav'
                convert_mp3_to_wav(file_path, output_wav_path)
                os.remove(file_path)  # Remove the original MP3 file
                file_path = output_wav_path  # Update path to the new WAV file
            except subprocess.CalledProcessError as e:
                print(f"FFmpeg conversion error: {e}")
                os.remove(file_path)
                return jsonify({"error": "Failed to convert MP3. Please check file integrity or try another file format."}), 400

        # Initialize Whisper model with GPU
        whisper_model = WhisperModel(model_name="large-v2")
        print(f"Running transcription on file: {file_path}")

        # Perform transcription on the GPU
        transcription = whisper_model.transcribe(file_path, language=language)

        # Save transcription to a file
        transcript_folder = current_app.config.get('TRANSCRIPT_FOLDER')
        os.makedirs(transcript_folder, exist_ok=True)
        base_filename = os.path.splitext(file.filename)[0] + "_transcription"
        transcript_filepath = os.path.join(transcript_folder, f"{base_filename}.txt")
        print(f"Saving transcription to: {transcript_filepath}")

        # Ensure unique file name if necessary
        counter = 1
        while os.path.exists(transcript_filepath):
            transcript_filepath = os.path.join(transcript_folder, f"{base_filename}_{counter}.txt")
            counter += 1

        with open(transcript_filepath, 'w', encoding='utf-8') as f:
            f.write(transcription)

        # Clean up GPU memory
        del whisper_model
        torch.cuda.empty_cache()

        print("Transcription complete, returning file.")
        return send_file(transcript_filepath, as_attachment=True, download_name=os.path.basename(transcript_filepath))

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        # Ensure cleanup of temporary files
        if os.path.exists(file_path):
            os.remove(file_path)

def convert_mp3_to_wav(input_path, output_path):
    command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-probesize', '50M',  # Extend probe size
        '-analyzeduration', '100M',  # Extend analysis duration
        '-i', input_path,  # Input file
        output_path  # Output file
    ]
    subprocess.run(command, check=True)

@main_blueprint.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200
