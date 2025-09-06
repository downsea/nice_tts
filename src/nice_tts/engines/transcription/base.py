"""Base classes for transcription engines.

This module defines the abstract interface that all transcription engines must implement,
providing a consistent API for different transcription providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import time

from ...core.config import TranscriptionConfig


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""
    
    text: str
    language: str
    confidence: float = 0.0
    processing_time: float = 0.0
    segments: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate result data."""
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if not isinstance(self.language, str):
            raise ValueError("language must be a string")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.processing_time < 0:
            raise ValueError("processing_time must be non-negative")


class TranscriptionEngine(ABC):
    """Abstract base class for transcription engines."""
    
    def __init__(self, config: TranscriptionConfig):
        """Initialize the transcription engine.
        
        Args:
            config: Transcription configuration
        """
        self.config = config
        self._model = None
        self._device = None
    
    @abstractmethod
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            TranscriptionResult: The transcription result
            
        Raises:
            TranscriptionError: If transcription fails
            FileNotFoundError: If audio file is not found
            UnsupportedAudioFormatError: If audio format is not supported
        """
        pass
    
    @abstractmethod
    def supports_language(self, language: str) -> bool:
        """Check if the engine supports a specific language.
        
        Args:
            language: Language code (e.g., 'en', 'zh')
            
        Returns:
            bool: True if language is supported
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats.
        
        Returns:
            List[str]: List of supported file extensions (e.g., ['.wav', '.mp3'])
        """
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the transcription model.
        
        Raises:
            ModelLoadError: If model fails to load
            ModelNotFoundError: If model is not found
        """
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload the transcription model to free memory."""
        pass
    
    def is_model_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dict with model information
        """
        return {
            "name": self.config.model_name,
            "language": self.config.language,
            "device": self._device,
            "loaded": self.is_model_loaded()
        }
    
    def validate_audio_file(self, audio_path: Path) -> bool:
        """Validate that an audio file can be processed.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            bool: True if file is valid
            
        Raises:
            FileNotFoundError: If file doesn't exist
            UnsupportedAudioFormatError: If format is not supported
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if not audio_path.is_file():
            raise ValueError(f"Path is not a file: {audio_path}")
        
        file_extension = audio_path.suffix.lower()
        if file_extension not in self.get_supported_formats():
            from ...core.exceptions import UnsupportedAudioFormatError
            raise UnsupportedAudioFormatError(
                f"Unsupported audio format: {file_extension}",
                audio_path=str(audio_path)
            )
        
        return True


class EngineRegistry:
    """Registry for transcription engines."""
    
    def __init__(self):
        self._engines: Dict[str, type] = {}
    
    def register(self, name: str, engine_class: type) -> None:
        """Register a transcription engine.
        
        Args:
            name: Engine name (e.g., 'whisper')
            engine_class: Engine class that implements TranscriptionEngine
        """
        if not issubclass(engine_class, TranscriptionEngine):
            raise ValueError(f"Engine class must inherit from TranscriptionEngine")
        
        self._engines[name] = engine_class
    
    def get(self, name: str) -> type:
        """Get a registered engine class.
        
        Args:
            name: Engine name
            
        Returns:
            Engine class
            
        Raises:
            KeyError: If engine is not registered
        """
        if name not in self._engines:
            raise KeyError(f"Unknown transcription engine: {name}")
        
        return self._engines[name]
    
    def list_engines(self) -> List[str]:
        """Get list of registered engine names."""
        return list(self._engines.keys())
    
    def create_engine(self, name: str, config: TranscriptionConfig) -> TranscriptionEngine:
        """Create an instance of a transcription engine.
        
        Args:
            name: Engine name
            config: Transcription configuration
            
        Returns:
            TranscriptionEngine instance
        """
        engine_class = self.get(name)
        return engine_class(config)


# Global registry instance
registry = EngineRegistry()