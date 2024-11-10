from flask import Flask, request, jsonify, send_file
import whisper
import os
import torch
import re
import unicodedata
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

# Initialize Flask app
app = Flask(__name__)

# Load the Whisper model
try:
    model = whisper.load_model("base")  # Choose model size based on your requirements
except Exception as e:
    print(f"Error loading Whisper model: {e}")

# Directory paths for saving files
UPLOAD_FOLDER = 'uploads'
TRANSCRIPT_FOLDER = 'transcripts'
FEEDBACK_FOLDER = 'feedback'
TRAINING_DATA_FOLDER = 'training_data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPT_FOLDER, exist_ok=True)
os.makedirs(FEEDBACK_FOLDER, exist_ok=True)
os.makedirs(TRAINING_DATA_FOLDER, exist_ok=True)

# Track the number of successful transcriptions
successful_transcriptions_count = 0
RETRAIN_THRESHOLD = 50  # Number of transcriptions before retraining

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    global successful_transcriptions_count
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        audio_file = request.files["file"]
        if audio_file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Get language parameter from form data
        language = request.form.get("language")
        supported_languages = ["en", "he", "ar", "ru", "de"]
        if language and language not in supported_languages:
            return jsonify({"error": f"Unsupported language. Supported languages are: {', '.join(supported_languages)}"}), 400

        file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        audio_file.save(file_path)

        print("Transcribing audio, please wait...")
        result = model.transcribe(file_path, language=language if language else None)
        
        transcription_text = result.get("text", "")
        if not transcription_text:
            return jsonify({"error": "Failed to transcribe audio"}), 500

        # Post-process Hebrew transcription if language is Hebrew
        if language == "he":
            transcription_text = normalize_hebrew_text(transcription_text)

        # Save the transcription word by word
        output_text_file = os.path.join(TRANSCRIPT_FOLDER, f"{os.path.splitext(audio_file.filename)[0]}_words.txt")
        with open(output_text_file, "w", encoding="utf-8") as file:
            file.write("\n".join(transcription_text.split()))

        # Save audio and transcription for future training in language-specific folder
        save_for_finetuning(audio_file.filename, transcription_text, language)
        
        successful_transcriptions_count += 1
        if successful_transcriptions_count >= RETRAIN_THRESHOLD:
            retrain_model(language=language)  # Retrain using the specific language

        os.remove(file_path)

        return send_file(output_text_file, as_attachment=True)

    except Exception as e:
        print(f"Error in transcription process: {e}")
        return jsonify({"error": "An error occurred during transcription"}), 500

@app.route("/feedback", methods=["POST"])
def receive_feedback():
    """Receive corrected transcription feedback from users to improve model accuracy."""
    try:
        data = request.json
        original_filename = data.get("original_filename")
        corrected_transcription = data.get("corrected_transcription")

        if not original_filename or not corrected_transcription:
            return jsonify({"error": "Filename and corrected transcription required"}), 400

        feedback_file = os.path.join(FEEDBACK_FOLDER, f"{original_filename}_feedback.txt")
        with open(feedback_file, "w", encoding="utf-8") as file:
            file.write(corrected_transcription)

        return jsonify({"message": "Feedback received"}), 200

    except Exception as e:
        print(f"Error in receiving feedback: {e}")
        return jsonify({"error": "An error occurred while receiving feedback"}), 500

def normalize_hebrew_text(text):
    """Custom function to normalize Hebrew text."""
    # Remove diacritics (Nikud) from Hebrew text
    text = ''.join([char for char in text if not unicodedata.category(char).startswith('Mn')])

    # Remove unnecessary whitespace and punctuation
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

    return text

def save_for_finetuning(filename, transcription, language):
    """Save audio-transcription pairs for future fine-tuning in language-specific folders."""
    language_folder = os.path.join(TRAINING_DATA_FOLDER, language)
    os.makedirs(language_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(language_folder, f"{filename}_{timestamp}.txt")
    with open(save_path, "w", encoding="utf-8") as file:
        file.write(transcription)

def retrain_model(language):
    """Fine-tune the model on saved data for the specified language."""
    global successful_transcriptions_count
    print(f"Retraining the model with new data for language: {language}...")

    # Load language-specific training data
    train_data = load_training_data(language)

    # Initialize optimizer and define training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()  # Adjust to appropriate loss for language model

    model.train()
    for epoch in range(5):  # Number of epochs can be adjusted
        for batch in train_data:
            audio, target_text = batch
            optimizer.zero_grad()
            output = model(audio)  # Placeholder, adapt based on model implementation
            loss = criterion(output, target_text)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    successful_transcriptions_count = 0
    print(f"Retraining completed and model updated for language: {language}")

class TextDataset(Dataset):
    """Custom dataset for loading text data."""
    def __init__(self, language_folder):
        self.data = []
        self.load_data(language_folder)

    def load_data(self, folder):
        """Load text files from a folder into the dataset."""
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read().strip()
                    self.data.append(text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_training_data(language):
    """Load training data for the specified language."""
    language_folder = os.path.join(TRAINING_DATA_FOLDER, language)

    # Check if language folder exists
    if not os.path.exists(language_folder):
        print(f"No training data found for language: {language}")
        return None

    # Create dataset and DataLoader
    dataset = TextDataset(language_folder)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    return data_loader

if __name__ == "__main__":
    app.run(debug=True)
