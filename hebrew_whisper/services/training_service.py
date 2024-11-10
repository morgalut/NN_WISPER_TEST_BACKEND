import os
import torch
import numpy as np
from scipy.io import wavfile
import whisper
from utilities.file_handler import FileHandler
import logging
from flask import current_app, Flask

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrainingService:
    def __init__(self, app=None):
        # Use Flask app's configuration for paths within application context
        if app is None:  # Allows for easier testing or non-web script usage
            app = current_app

        with app.app_context():
            self.model_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], 'model.pt')
            self.training_data_folder = app.config['TRAINING_DATA_FOLDER']
            self.model = self.load_model()

    def load_model(self):
        """
        Load an existing model or initialize a new one if it does not exist.
        """
        if os.path.exists(self.model_path):
            logging.info(f"Loading model from {self.model_path}")
            return torch.load(self.model_path)
        else:
            logging.info("Initializing new Whisper model.")
            return whisper.load_model("base")

    def save_model(self):
        """
        Save the trained model to a file.
        """
        torch.save(self.model, self.model_path)
        logging.info(f"Model saved to {self.model_path}")

    def train_model(self, feedbacks):
        """
        Train the model using provided feedback.
        """
        for feedback in feedbacks:
            try:
                audio_path = feedback['audio_path']
                correct_transcript = feedback['correct_transcript']
                audio_data, rate = wavfile.read(audio_path)
                mel_spec = self.preprocess_audio(audio_data, rate)
                loss = self.model.train_on_batch(mel_spec, correct_transcript)
                logging.info(f"Training loss: {loss}")
            except Exception as e:
                logging.error(f"Error during model training: {e}")

        self.save_model()

    def preprocess_audio(self, audio_data, rate):
        """
        Preprocess audio data to mel spectrogram.
        """
        if audio_data is not None:
            mel_spec = np.log1p(np.abs(np.fft.rfft(audio_data)))
            return torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
        else:
            logging.error("Invalid audio data received for preprocessing.")
            return None

    def retrain_with_feedback(self):
        """
        Check for new feedback and retrain the model if necessary.
        """
        feedback_files = FileHandler.check_new_files(self.training_data_folder, ('.txt',))
        feedbacks = []
        for file in feedback_files:
            try:
                with open(os.path.join(self.training_data_folder, file), 'r') as f:
                    feedback = {'audio_path': file.replace('_transcript.txt', '.wav'),
                                'correct_transcript': f.read()}
                    feedbacks.append(feedback)
            except Exception as e:
                logging.error(f"Error reading feedback file {file}: {e}")
                
        if feedbacks:
            self.train_model(feedbacks)
        else:
            logging.info("No new feedback to train on.")
