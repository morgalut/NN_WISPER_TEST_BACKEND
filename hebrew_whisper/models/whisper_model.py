import whisper
import torch
import torchaudio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.model = whisper.load_model(model_name).to(self.device)

        # Enable mixed precision for faster computation on CUDA
        if self.device.type == "cuda":
            self.model = self.model.half()

        self.beam_size = beam_size
        self.temperature = temperature

    def preprocess_audio(self, audio_path):
        """
        Preprocess audio for the Whisper model, ensuring it's on the target device.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            torch.Tensor: Preprocessed audio tensor on the appropriate device.
        """
        logger.info(f"Preprocessing audio: {audio_path}")

        audio, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            logger.info(f"Resampling audio from {sample_rate} Hz to 16000 Hz.")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio = resampler(audio)

        # Normalize audio to [-1, 1] and move it to the target device
        audio = audio / torch.max(torch.abs(audio))
        logger.info("Audio preprocessing completed.")
        return audio.to(self.device)

    def transcribe(self, audio_path, language="he"):
        """
        Transcribe audio using the Whisper model, optimized for GPU usage.

        Args:
            audio_path (str): Path to the audio file.
            language (str): Language code to skip detection.
            
        Returns:
            str: Transcribed text.
        """
        logger.info(f"Starting transcription on {self.device}")

        # Preprocess and transcribe audio
        audio = self.preprocess_audio(audio_path)

        # Run transcription
        result = self.model.transcribe(
            audio_path, language=language, beam_size=self.beam_size, temperature=self.temperature
        )

        logger.info("Transcription completed.")
        return result.get("text")

    def clean_up(self):
        """
        Explicitly release GPU memory to reduce memory consumption.
        """
        logger.info("Cleaning up model and freeing GPU memory.")
        del self.model
        torch.cuda.empty_cache()

    def batch_transcribe(self, audio_paths, language="he"):
        """
        Batch transcription to handle multiple files while optimizing GPU usage.
        
        Args:
            audio_paths (list): List of audio file paths.
            language (str): Language code to skip detection.
            
        Returns:
            dict: Dictionary mapping file paths to their transcriptions.
        """
        logger.info("Starting batch transcription.")
        transcriptions = {}

        for audio_path in audio_paths:
            try:
                logger.info(f"Transcribing file: {audio_path}")
                transcriptions[audio_path] = self.transcribe(audio_path, language)
            except Exception as e:
                logger.error(f"Error transcribing {audio_path}: {e}")
                transcriptions[audio_path] = None

        logger.info("Batch transcription completed.")
        return transcriptions
