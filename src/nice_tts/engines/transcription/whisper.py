"""Whisper transcription engine implementation.

This module provides a Whisper-based transcription engine that supports
GPU acceleration and various model sizes.
"""

import re
import time
import torch
import whisper
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import TranscriptionEngine, TranscriptionResult, registry
from ...core.config import TranscriptionConfig
from ...core.exceptions import (
    ModelLoadError,
    ModelNotFoundError,
    TranscriptionFailureError,
    DeviceError,
    AudioProcessingError,
    AudioCorruptedError
)

# Import M4A utilities
try:
    from ...utils.m4a_validator import M4AValidator
    from ...utils.m4a_preprocessor import M4APreprocessor
    M4A_SUPPORT_AVAILABLE = True
except ImportError:
    M4A_SUPPORT_AVAILABLE = False
    M4AValidator = None
    M4APreprocessor = None

import logging
logger = logging.getLogger('nice_tts')


class WhisperEngine(TranscriptionEngine):
    """Whisper-based transcription engine."""
    
    # Supported Whisper models
    SUPPORTED_MODELS = [
        "tiny", "tiny.en", "base", "base.en", "small", "small.en",
        "medium", "medium.en", "large", "large-v1", "large-v2", 
        "large-v3", "large-v3-turbo"
    ]
    
    # Supported audio formats
    SUPPORTED_FORMATS = [".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"]
    
    # Supported languages (ISO 639-1 codes)
    SUPPORTED_LANGUAGES = [
        "zh", "en", "es", "fr", "de", "it", "ja", "ko", "pt", "ru",
        "ar", "hi", "th", "vi", "id", "ms", "tl", "nl", "sv", "no",
        "da", "fi", "pl", "cs", "sk", "hu", "ro", "bg", "hr", "sl",
        "et", "lv", "lt", "mt", "ga", "cy", "eu", "ca", "gl", "ast"
    ]
    
    def __init__(self, config: TranscriptionConfig):
        """Initialize Whisper engine.
        
        Args:
            config: Transcription configuration
        """
        super().__init__(config)
        self._validate_config()
    
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe an audio file using Whisper.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            TranscriptionResult: The transcription result
        """
        # Validate input
        self.validate_audio_file(audio_path)
        
        # Preprocess M4A files if needed
        processed_audio_path = audio_path
        temp_files_to_cleanup = []
        
        try:
            # Special handling for M4A files
            if audio_path.suffix.lower() == '.m4a' and M4A_SUPPORT_AVAILABLE:
                processed_audio_path = self._handle_m4a_file(audio_path)
                if processed_audio_path != audio_path:
                    temp_files_to_cleanup.append(processed_audio_path)
            
            # Ensure model is loaded
            if not self.is_model_loaded():
                self.load_model()
            
            start_time = time.time()
            
            try:
                # Prepare transcription options
                options = self._get_transcription_options()
                
                # Perform transcription with enhanced error handling
                result = self._safe_transcribe(str(processed_audio_path), **options)
                
                processing_time = time.time() - start_time
                
                # Extract transcription text
                raw_text = result.get("text", "").strip()
                language = result.get("language", self.config.language)
                
                # Get segments for formatting
                segments = result.get("segments", [])
                
                # Format the transcript text based on config
                if self.config.format_output:
                    if self.config.line_break_mode == "segment":
                        # Use segment-based formatting
                        formatted_text = self.format_transcript_text(segments, raw_text)
                    else:
                        # Use sentence-based formatting (default)
                        formatted_text = self._format_from_raw_text(raw_text)
                else:
                    # No formatting, use raw text
                    formatted_text = raw_text
                
                # Calculate confidence (Whisper doesn't provide per-text confidence)
                confidence = self._calculate_confidence(segments)
                
                return TranscriptionResult(
                    text=formatted_text,
                    language=language,
                    confidence=confidence,
                    processing_time=processing_time,
                    segments=segments,
                    metadata={
                        "model": self.config.model_name,
                        "device": str(self._device),
                        "audio_duration": result.get("duration", 0),
                        "temperature": self.config.temperature,
                        "raw_text": raw_text,  # Keep original for reference
                        "original_file": str(audio_path),
                        "processed_file": str(processed_audio_path)
                    }
                )
                
            except Exception as e:
                processing_time = time.time() - start_time
                raise TranscriptionFailureError(
                    f"Whisper transcription failed: {e}",
                    model_name=self.config.model_name,
                    details={
                        "audio_path": str(audio_path),
                        "processed_path": str(processed_audio_path),
                        "processing_time": processing_time,
                        "error_type": type(e).__name__
                    }
                ) from e
                
        finally:
            # Clean up temporary files
            for temp_file in temp_files_to_cleanup:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")
    
    def _handle_m4a_file(self, audio_path: Path) -> Path:
        """Handle M4A file with validation and preprocessing.
        
        Args:
            audio_path: Path to the M4A file
            
        Returns:
            Path: Path to the file to use for transcription (may be preprocessed)
        """
        if not M4A_SUPPORT_AVAILABLE:
            logger.warning("M4A support not available - processing with default Whisper")
            return audio_path
        
        # Validate the M4A file
        logger.debug(f"Validating M4A file: {audio_path}")
        validation_result = M4AValidator.validate_m4a(audio_path)
        
        if validation_result.is_valid:
            logger.debug(f"M4A file {audio_path} is valid, proceeding with normal processing")
            return audio_path
        
        # Log validation issues
        logger.warning(f"M4A file {audio_path} validation issues: {validation_result.issues}")
        
        # Handle based on validation result
        if not validation_result.can_be_processed:
            # Try to repair the file
            logger.info(f"Attempting to repair M4A file: {audio_path}")
            repaired_path = M4APreprocessor.create_temp_repaired_file(audio_path)
            if repaired_path and repaired_path.exists():
                # Validate the repaired file
                repair_validation = M4AValidator.validate_m4a(repaired_path)
                if repair_validation.is_valid or repair_validation.can_be_processed:
                    logger.info(f"Successfully repaired M4A file: {audio_path}")
                    return repaired_path
                else:
                    logger.warning(f"Repaired M4A file still has issues: {repair_validation.issues}")
            
            # If repair failed, try conversion to WAV
            logger.info(f"Converting M4A to WAV as fallback: {audio_path}")
            wav_path = M4APreprocessor.create_temp_wav_file(audio_path)
            if wav_path and wav_path.exists():
                logger.info(f"Successfully converted M4A to WAV: {audio_path}")
                return wav_path
            else:
                # If all else fails, raise an error
                raise AudioCorruptedError(
                    f"M4A file is corrupted and cannot be processed: {audio_path}",
                    audio_path=str(audio_path)
                )
        else:
            # File can be processed but has issues, try preprocessing
            logger.info(f"Preprocessing M4A file to fix issues: {audio_path}")
            repaired_path = M4APreprocessor.create_temp_repaired_file(audio_path)
            if repaired_path and repaired_path.exists():
                return repaired_path
            else:
                # Fall back to original file
                logger.warning(f"Could not preprocess M4A file, using original: {audio_path}")
                return audio_path
    
    def _safe_transcribe(self, audio_path: str, **options) -> Dict[str, Any]:
        """Perform transcription with enhanced error handling.
        
        Args:
            audio_path: Path to audio file
            **options: Transcription options
            
        Returns:
            Dict: Transcription result
            
        Raises:
            TranscriptionFailureError: If transcription fails
        """
        try:
            result = self._model.transcribe(audio_path, **options)
            return result
        except ZeroDivisionError as e:
            # Handle the specific ZeroDivisionError for M4A files
            logger.error(f"ZeroDivisionError during transcription of {audio_path}: {e}")
            
            # If this is an M4A file, provide specific handling
            if audio_path.lower().endswith('.m4a'):
                logger.info(f"Attempting to handle ZeroDivisionError for M4A file: {audio_path}")
                # Try to convert to WAV and retry
                if M4A_SUPPORT_AVAILABLE:
                    try:
                        wav_path = M4APreprocessor.create_temp_wav_file(Path(audio_path))
                        if wav_path and wav_path.exists():
                            logger.info(f"Retrying transcription with WAV conversion: {wav_path}")
                            try:
                                result = self._model.transcribe(str(wav_path), **options)
                                # Clean up temporary file
                                try:
                                    wav_path.unlink()
                                except Exception:
                                    pass
                                return result
                            except Exception as retry_e:
                                logger.error(f"Retry with WAV conversion also failed: {retry_e}")
                                # Clean up temporary file
                                try:
                                    wav_path.unlink()
                                except Exception:
                                    pass
                                raise TranscriptionFailureError(
                                    f"Transcription failed after WAV conversion retry: {retry_e}",
                                    model_name=self.config.model_name
                                ) from retry_e
                    except Exception as conversion_error:
                        logger.error(f"Failed to convert M4A to WAV: {conversion_error}")
            
            # If we get here, re-raise the original error
            raise TranscriptionFailureError(
                f"ZeroDivisionError during transcription: {e}",
                model_name=self.config.model_name
            ) from e
        except Exception as e:
            # Re-raise other exceptions
            raise e
    
    def supports_language(self, language: str) -> bool:
        """Check if Whisper supports a specific language."""
        return language.lower() in self.SUPPORTED_LANGUAGES
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats."""
        return self.SUPPORTED_FORMATS.copy()
    
    def load_model(self) -> None:
        """Load the Whisper model."""
        try:
            # Detect and set device
            self._device = self._detect_device()
            
            # Load model
            model_name = self.config.model_name
            if model_name not in self.SUPPORTED_MODELS:
                raise ModelNotFoundError(
                    f"Unsupported Whisper model: {model_name}",
                    model_name=model_name,
                    details={"supported_models": self.SUPPORTED_MODELS}
                )
            
            # Load with custom cache directory if specified
            download_root = self.config.cache_dir if self.config.cache_dir else None
            
            self._model = whisper.load_model(
                model_name,
                device=self._device,
                download_root=download_root
            )
            
            # Move model to device if needed
            if self._device != "cpu":
                self._model = self._model.to(self._device)
            
        except Exception as e:
            self._model = None
            self._device = None
            
            if isinstance(e, (ModelNotFoundError, DeviceError)):
                raise
            else:
                raise ModelLoadError(
                    f"Failed to load Whisper model '{self.config.model_name}': {e}",
                    model_name=self.config.model_name,
                    details={"error_type": type(e).__name__}
                ) from e
    
    def unload_model(self) -> None:
        """Unload the Whisper model to free memory."""
        if self._model is not None:
            # Clear CUDA cache if using GPU
            if self._device and "cuda" in str(self._device):
                torch.cuda.empty_cache()
            
            del self._model
            self._model = None
    
    def _validate_config(self) -> None:
        """Validate the configuration for Whisper."""
        # Check model name
        if self.config.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported Whisper model: {self.config.model_name}. "
                f"Supported models: {self.SUPPORTED_MODELS}"
            )
        
        # Check language
        if not self.supports_language(self.config.language):
            raise ValueError(
                f"Unsupported language: {self.config.language}. "
                f"Supported languages: {self.SUPPORTED_LANGUAGES}"
            )
        
        # Check cache directory if specified
        if self.config.cache_dir:
            cache_path = Path(self.config.cache_dir)
            try:
                cache_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Cannot create cache directory: {e}")
    
    def _detect_device(self) -> str:
        """Detect the best available device for computation."""
        if self.config.device == "cpu":
            return "cpu"
        elif self.config.device == "cuda":
            if not torch.cuda.is_available():
                raise DeviceError(
                    "CUDA device requested but not available",
                    details={"requested_device": "cuda", "cuda_available": False}
                )
            return "cuda"
        elif self.config.device == "auto":
            # Auto-detect best device
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        else:
            raise DeviceError(
                f"Unsupported device: {self.config.device}",
                details={"supported_devices": ["cpu", "cuda", "auto"]}
            )
    
    def _get_transcription_options(self) -> Dict[str, Any]:
        """Get Whisper transcription options from config."""
        options = {
            "language": self.config.language if self.config.language != "auto" else None,
            "temperature": self.config.temperature,
            "best_of": self.config.best_of,
            "beam_size": self.config.beam_size,
            "verbose": False,  # We handle our own logging
        }
        
        # Remove None values
        return {k: v for k, v in options.items() if v is not None}
    
    def _calculate_confidence(self, segments: List[Dict[str, Any]]) -> float:
        """Calculate average confidence from segments.
        
        Whisper provides per-token probabilities in segments, we can use these
        to estimate overall confidence.
        """
        if not segments:
            return 0.0
        
        total_confidence = 0.0
        total_tokens = 0
        
        for segment in segments:
            # Some versions of Whisper include token-level data
            tokens = segment.get("tokens", [])
            if tokens and "avg_logprob" in segment:
                # Convert log probability to confidence (approximate)
                confidence = min(1.0, max(0.0, (segment["avg_logprob"] + 1.0)))
                total_confidence += confidence * len(tokens)
                total_tokens += len(tokens)
            else:
                # Fallback: assume reasonable confidence
                total_confidence += 0.8 * len(segment.get("text", "").split())
                total_tokens += len(segment.get("text", "").split())
        
        if total_tokens == 0:
            return 0.0
        
        return total_confidence / total_tokens
    
    @classmethod
    def check_gpu_availability(cls) -> Dict[str, Any]:
        """Check GPU availability and return diagnostic information."""
        info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": 0,
            "devices": []
        }
        
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["device_count"] = torch.cuda.device_count()
            
            for i in range(torch.cuda.device_count()):
                device_info = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                    "memory_reserved": torch.cuda.memory_reserved(i),
                    "memory_allocated": torch.cuda.memory_allocated(i)
                }
                info["devices"].append(device_info)
        
        return info
    
    def _format_from_raw_text(self, text: str) -> str:
        """Enhanced sentence-based formatting for Chinese text.
        
        Args:
            text: Raw transcript text
            
        Returns:
            str: Formatted text with improved sentence boundaries
        """
        import re
        
        # Clean up the text first
        text = text.strip()
        if not text:
            return ""
        
        # Enhanced Chinese sentence splitting pattern
        # Include various Chinese punctuation marks and western equivalents
        sentence_endings = r'([。！？；.!?;](?:["''""\)\]]*)?\s*)'
        
        # Split while preserving delimiters
        parts = re.split(sentence_endings, text)
        
        sentences = []
        current_sentence = ""
        
        for i, part in enumerate(parts):
            if not part.strip():
                continue
                
            # Check if this part is a sentence ending
            if re.match(sentence_endings, part):
                current_sentence += part
                # Clean up the sentence
                clean_sentence = self._clean_sentence(current_sentence)
                if clean_sentence:
                    sentences.append(clean_sentence)
                current_sentence = ""
            else:
                current_sentence += part
        
        # Handle any remaining text
        if current_sentence.strip():
            clean_sentence = self._clean_sentence(current_sentence)
            if clean_sentence:
                sentences.append(clean_sentence)
        
        return "\n".join(sentences)
    
    def _clean_sentence(self, sentence: str) -> str:
        """Clean up individual sentences.
        
        Args:
            sentence: Raw sentence text
            
        Returns:
            str: Cleaned sentence
        """
        # Remove extra whitespace
        sentence = re.sub(r'\s+', ' ', sentence)
        
        # Remove leading/trailing whitespace
        sentence = sentence.strip()
        
        # Remove common speech disfluencies and filler words
        fillers = ['嗯', '呃', '啊', '呀', '哦', '额', '那个', '这个', '就是说', '然后']
        
        # Only remove fillers at the beginning of sentences or when isolated
        for filler in fillers:
            # Remove at start of sentence
            pattern = rf'^{re.escape(filler)}[，,\s]*'
            sentence = re.sub(pattern, '', sentence)
            
            # Remove when isolated with punctuation
            pattern = rf'[，,\s]+{re.escape(filler)}[，,\s]+'
            sentence = re.sub(pattern, '，', sentence)
        
        return sentence


# Register the Whisper engine
registry.register("whisper", WhisperEngine)