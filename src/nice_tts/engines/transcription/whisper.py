"""Whisper transcription engine implementation.

This module provides a Whisper-based transcription engine that supports
GPU acceleration and various model sizes.
"""

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
    AudioProcessingError
)


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
        
        # Ensure model is loaded
        if not self.is_model_loaded():
            self.load_model()
        
        start_time = time.time()
        
        try:
            # Prepare transcription options
            options = self._get_transcription_options()
            
            # Perform transcription
            result = self._model.transcribe(str(audio_path), **options)
            
            processing_time = time.time() - start_time
            
            # Extract transcription text
            text = result.get("text", "").strip()
            language = result.get("language", self.config.language)
            
            # Calculate confidence (Whisper doesn't provide per-text confidence)
            segments = result.get("segments", [])
            confidence = self._calculate_confidence(segments)
            
            return TranscriptionResult(
                text=text,
                language=language,
                confidence=confidence,
                processing_time=processing_time,
                segments=segments,
                metadata={
                    "model": self.config.model_name,
                    "device": str(self._device),
                    "audio_duration": result.get("duration", 0),
                    "temperature": self.config.temperature
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            raise TranscriptionFailureError(
                f"Whisper transcription failed: {e}",
                model_name=self.config.model_name,
                details={
                    "audio_path": str(audio_path),
                    "processing_time": processing_time,
                    "error_type": type(e).__name__
                }
            ) from e
    
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


# Register the Whisper engine
registry.register("whisper", WhisperEngine)