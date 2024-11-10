from flask import jsonify, request, current_app, send_file
from models.whisper_model import WhisperModel
from utilities.file_handler import FileHandler
from utilities.text_normalizer import AudioPreprocessor, TextNormalizer
import os

class TranscriptionService:
    @staticmethod
    def transcribe(request=None, file_path=None):
        if request:
            audio_file = request.files.get('file')
            if not audio_file:
                return jsonify({"error": "No file uploaded"}), 400
            temp_path = os.path.join(current_app.config['UPLOAD_FOLDER'], audio_file.filename)
            audio_file.save(temp_path)
        elif file_path:
            temp_path = file_path
        else:
            return jsonify({"error": "No audio file provided"}), 400

        try:
            preprocessor = AudioPreprocessor('base', current_app.config['TRAINING_DATA_FOLDER'])
            processed_audio, rate = preprocessor.preprocess_audio(temp_path)

            whisper_model = WhisperModel()
            transcription_result = whisper_model.transcribe(processed_audio, rate)

            # Ensure Hebrew language processing if detected or specified
            language = request.form.get("language", "he")
            normalized_text = TextNormalizer.normalize_text(transcription_result, language)

            transcript_path = os.path.join(current_app.config['TRANSCRIPT_FOLDER'], f"{os.path.splitext(audio_file.filename)[0]}_transcript.txt")
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(normalized_text)

            if request:
                os.remove(temp_path)

            return send_file(transcript_path, as_attachment=True)
        except Exception as e:
            if request:
                os.remove(temp_path)
            return jsonify({"error": str(e)}), 500
