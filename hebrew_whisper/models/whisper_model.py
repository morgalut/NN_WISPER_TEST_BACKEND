# C:\Users\Mor\Desktop\NN_Whisper_AI_Flask\hebrew_whisper\models\whisper_model.py

import asyncio
import whisper
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperModel:
    def __init__(self, model_size='large', beam_size=5):
        """
        Initialize the Whisper model with a specified size and decoding options.
        
        Args:
            model_size (str): The size of the Whisper model (e.g., 'base', 'small', 'medium', 'large').
            beam_size (int): Beam search width for improved decoding accuracy.
        """
        self.model_size = model_size
        self.beam_size = beam_size
        self.model_future = None  # Async task to load model
        self.model = None

    def initialize_model(self):
        """
        Initialize the Whisper model asynchronously if not already done.
        """
        if self.model_future is None:
            self.model_future = asyncio.create_task(self.async_load_model(self.model_size))

    @staticmethod
    async def async_load_model(model_size):
        """
        Load the Whisper model asynchronously in a separate thread.
        
        Args:
            model_size (str): The size of the Whisper model to load.
            
        Returns:
            model: Loaded Whisper model.
        """
        try:
            logger.info(f"Loading Whisper model: {model_size}")
            model = await asyncio.to_thread(whisper.load_model, model_size)
            logger.info(f"Model loaded successfully: {model_size}")
            return model
        except whisper.WhisperException as whisper_error:
            logger.error(f"Whisper-specific error during model load: {whisper_error}")
            raise
        except Exception as general_error:
            logger.error(f"General error during model load: {general_error}")
            raise

    async def get_model(self):
        """
        Ensure the Whisper model is loaded and ready for use.
        
        Returns:
            model: Loaded Whisper model instance.
        """
        if self.model_future is None:
            self.initialize_model()
        if not self.model:
            self.model = await self.model_future
        return self.model

    async def transcribe(self, audio_path, language="he"):
        """
        Transcribe the given audio file using the Whisper model.
        
        Args:
            audio_path (str): Path to the audio file.
            language (str): Language code for transcription, defaults to Hebrew ('he').
            
        Returns:
            str: Transcribed text from the audio.
        """
        model = await self.get_model()  # Ensure model is loaded
        # Use beam search for potentially more accurate decoding
        result = await asyncio.to_thread(
            model.transcribe, audio_path, language=language, beam_size=self.beam_size
        )
        return result.get("text")
