# C:\Users\Mor\Desktop\NN_Whisper_AI_Flask\NN_Whisper_AI_Flask_backend\hebrew_whisper\models\whisper_model.py

import asyncio
import whisper
import torch
import torchaudio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperModel:
    def __init__(self, model_name="large-v2", beam_size=3, temperature=0.3):
        """
        Initialize the Whisper model with GPU support, mixed precision, and configurable parameters.
        
        Args:
            model_name (str): The model variant to load, e.g., 'large-v2'.
            beam_size (int): Beam search width for decoding.
            temperature (float): Temperature for randomness in decoding.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_name).to(self.device)
        
        # Only apply half precision if CUDA is available
        if self.device == "cuda":
            self.model = self.model.half()
        
        self.beam_size = beam_size
        self.temperature = temperature


    def preprocess_audio(self, audio_path):
        """
        Load and resample audio to 16 kHz for optimal processing with Whisper.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            torch.Tensor: Processed audio tensor.
        """
        audio, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio = resampler(audio)
        return audio

    async def transcribe_async(self, audio_path, language="en"):
        """
        Asynchronously transcribe the given audio file using Whisper.
        
        Args:
            audio_path (str): Path to the audio file.
            language (str): Language code to skip language detection.
            
        Returns:
            str: Transcribed text.
        """
        audio = self.preprocess_audio(audio_path)
        result = await asyncio.to_thread(
            self.model.transcribe,
            audio_path,
            language=language,
            beam_size=self.beam_size,
            temperature=self.temperature
        )
        return result.get("text")

    def transcribe(self, audio_input, language="he"):
        """
        Transcribe the audio input using Whisper.
        
        Args:
            audio_input (str or torch.Tensor): Path to the audio file or preprocessed audio tensor.
            language (str): Language code to skip detection.
            
        Returns:
            str: Transcribed text.
        """
        # If audio_input is a file path, load and preprocess it
        if isinstance(audio_input, str):
            # Pass the file path directly
            result = self.model.transcribe(audio_input, language=language, beam_size=self.beam_size, temperature=self.temperature)
        elif isinstance(audio_input, torch.Tensor):
            # Handle preprocessed audio tensor
            result = self.model.transcribe(audio_input, language=language, beam_size=self.beam_size, temperature=self.temperature)
        else:
            raise ValueError("Invalid audio input type. Expected a file path or torch.Tensor.")
        
        return result.get("text")


