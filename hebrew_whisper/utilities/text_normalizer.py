# C:\Users\Mor\Desktop\NN_Whisper_AI_Flask\hebrew_whisper\utilities\text_normalizer.py

import os
import re
import numpy as np
import soundfile as sf
import whisper
import unicodedata
from scipy.signal import butter, lfilter
from flask import current_app

class AudioPreprocessor:
    def __init__(self, model_path='base', training_data_folder=None):
        """
        Initialize the AudioPreprocessor with model path and training data folder.
        
        Args:
            model_path (str): Path to the Whisper model or model size.
            training_data_folder (str): Path to the folder with training data.
        """
        self.model_path = model_path
        self.training_data_folder = training_data_folder
        self.model = self.load_model()

    def load_model(self):
        return whisper.load_model(self.model_path)

    def preprocess_audio(self, audio_path):
        """
        Preprocess audio by applying noise reduction and sample rate normalization.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            tuple: Processed audio data and sample rate.
        """
        data, rate = sf.read(audio_path)
        # Apply noise reduction (optional, based on project needs)
        data = self.noise_reduction(data, rate)
        return data, rate

    def noise_reduction(self, data, rate):
        """
        Apply a basic noise reduction filter to the audio data.
        
        Args:
            data (ndarray): Audio data array.
            rate (int): Sample rate of the audio data.
            
        Returns:
            ndarray: Noise-reduced audio data.
        """
        nyquist = 0.5 * rate
        low = 300 / nyquist  # Low-frequency cut-off for reducing noise
        high = 3000 / nyquist  # High-frequency cut-off for reducing noise
        b, a = butter(1, [low, high], btype='band')  # Create bandpass filter
        return lfilter(b, a, data)

class TextNormalizer:
    @staticmethod
    def normalize_text(text, language):
        """
        Normalize text based on language-specific requirements.
        
        Args:
            text (str): The text to normalize.
            language (str): Language code, defaults to Hebrew ('he').
            
        Returns:
            str: Normalized text.
        """
        if language == "he":
            return TextNormalizer.normalize_hebrew_text(text)
        return text  # Optionally handle other languages or return raw text

    @staticmethod
    def normalize_hebrew_text(text):
        """
        Normalize Hebrew text by removing diacritics and special characters.
        
        Args:
            text (str): The Hebrew text to normalize.
            
        Returns:
            str: Cleaned Hebrew text.
        """
        # Remove Hebrew diacritics
        text = ''.join(char for char in text if not unicodedata.category(char).startswith('Mn'))
        # Remove non-alphanumeric characters
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def correct_text(text):
        """
        Apply basic spell-check and corrections to improve transcription accuracy.
        
        Args:
            text (str): Transcribed text to correct.
            
        Returns:
            str: Corrected text.
        """
        # Example correction step for Hebrew; can be replaced with spell-check logic
        corrections = {"עיר נמל": "עִיר נָמֵל"}  # Add common phrases or corrections here
        for incorrect, correct in corrections.items():
            text = text.replace(incorrect, correct)
        return text
