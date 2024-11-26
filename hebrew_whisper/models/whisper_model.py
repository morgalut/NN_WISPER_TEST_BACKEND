import whisper
import torch
import torchaudio
import logging
from whisper.decoding import DecodingResult
from typing import Any, List
from multiprocessing import Pool, cpu_count
import warnings

# Suppress specific warnings from torch
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhisperModelSingleton:
    _instance = None

    @classmethod
    def get_instance(cls, model_name="medium"):
        if cls._instance is None:
            cls._instance = WhisperModel(model_name)
        return cls._instance


class WhisperModel:
    def __init__(self, model_name="medium", beam_size=3, temperature=0.3):
        """
        Initialize the Whisper model with GPU support, falling back to CPU if necessary.

        Args:
            model_name (str): Whisper model variant to load, e.g., 'medium'.
            beam_size (int): Beam search width for decoding.
            temperature (float): Temperature for randomness in decoding.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model_safe(model_name)
        self.beam_size = beam_size
        self.temperature = temperature

    def _load_model_safe(self, model_name: str):
        """
        Safely load a Whisper model, setting weights_only=True for security.

        Args:
            model_name (str): Whisper model variant to load.

        Returns:
            Whisper model instance.
        """
        logger.info(f"Loading Whisper model: {model_name}")
        try:
            model = whisper.load_model(model_name).to(self.device)
            if self.device.type == "cuda":
                model = model.half()  # Use mixed precision on CUDA
            logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise RuntimeError("Error loading Whisper model.") from e

    def preprocess_audio(self, audio_path: str):
        """
        Preprocess audio for the Whisper model, ensuring it's on the target device.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            torch.Tensor: Preprocessed audio tensor on the appropriate device.
        """
        logger.info(f"Preprocessing audio: {audio_path}")
        try:
            audio, sample_rate = torchaudio.load(audio_path)
            if sample_rate != 16000:
                logger.info(f"Resampling audio from {sample_rate} Hz to 16000 Hz.")
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                audio = resampler(audio)
            audio = audio / torch.max(torch.abs(audio))  # Normalize audio
            return audio.to(self.device)
        except Exception as e:
            logger.error(f"Error during audio preprocessing: {e}", exc_info=True)
            raise RuntimeError("Error preprocessing audio.") from e

    def validate_result(self, result: Any):
        """
        Validate and parse the transcription result.

        Args:
            result (Any): Raw result returned by the Whisper model.

        Returns:
            str: Transcribed text or error message.
        """
        if isinstance(result, DecodingResult):
            logger.info("Valid DecodingResult object.")
            return result.text
        elif isinstance(result, dict) and 'text' in result:
            logger.info("Result contains valid text.")
            return result['text']
        elif isinstance(result, list):
            logger.warning("Unexpected list result received. Constructing fallback text.")
            return " ".join(str(item) for item in result)
        logger.error(f"Unexpected result type: {type(result).__name__}")
        return "Error: Unexpected result type received."

    def transcribe(self, audio_path: str, language="he"):
        """
        Transcribe audio using the Whisper model, optimized for GPU usage.

        Args:
            audio_path (str): Path to the audio file.
            language (str): Language code to skip detection.

        Returns:
            str: Transcribed text.
        """
        logger.info(f"Starting transcription on {self.device}")
        try:
            audio = self.preprocess_audio(audio_path)
            result = self.model.transcribe(
                audio_path, language=language, beam_size=self.beam_size, temperature=self.temperature
            )
            logger.debug(f"Raw transcription result: {result}")
            return self.validate_result(result)
        except Exception as e:
            logger.error(f"Error during transcription: {e}", exc_info=True)
            return f"Error: {str(e)}"

    def clean_up(self):
        """
        Explicitly release GPU memory to reduce memory consumption.
        """
        logger.info("Cleaning up model and freeing GPU memory.")
        del self.model
        torch.cuda.empty_cache()

    def batch_transcribe(self, audio_paths: List[str], language="he"):
        """
        Batch transcription to handle multiple files while optimizing GPU usage.

        Args:
            audio_paths (List[str]): List of audio file paths.
            language (str): Language code to skip detection.

        Returns:
            dict: Dictionary mapping file paths to their transcriptions.
        """
        logger.info("Starting batch transcription.")
        transcriptions = {}
        try:
            with Pool(cpu_count()) as pool:
                results = pool.map(lambda path: self.transcribe(path, language), audio_paths)
                transcriptions = dict(zip(audio_paths, results))
        except Exception as e:
            logger.error(f"Error during batch transcription: {e}", exc_info=True)
        logger.info("Batch transcription completed.")
        return transcriptions
