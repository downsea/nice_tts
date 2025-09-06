"""Test transcription engine functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import torch

from nice_tts.engines.transcription.base import (
    TranscriptionEngine, TranscriptionResult, EngineRegistry, registry
)
from nice_tts.engines.transcription.whisper import WhisperEngine
from nice_tts.core.config import TranscriptionConfig
from nice_tts.core.exceptions import (
    ModelLoadError, ModelNotFoundError, TranscriptionFailureError,
    DeviceError, UnsupportedAudioFormatError
)


class TestTranscriptionResult:
    """Test TranscriptionResult class."""
    
    def test_valid_result(self):
        """Test creating valid transcription result."""
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            confidence=0.95,
            processing_time=1.5
        )
        
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.confidence == 0.95
        assert result.processing_time == 1.5
        
    def test_invalid_confidence(self):
        """Test invalid confidence values."""
        with pytest.raises(ValueError):
            TranscriptionResult(
                text="test",
                language="en",
                confidence=1.5  # > 1.0
            )
            
        with pytest.raises(ValueError):
            TranscriptionResult(
                text="test",
                language="en",
                confidence=-0.1  # < 0.0
            )
            
    def test_invalid_processing_time(self):
        """Test invalid processing time."""
        with pytest.raises(ValueError):
            TranscriptionResult(
                text="test",
                language="en",
                processing_time=-1.0
            )


class MockTranscriptionEngine(TranscriptionEngine):
    """Mock transcription engine for testing."""
    
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        return TranscriptionResult(
            text="Mock transcription",
            language="en",
            confidence=0.9,
            processing_time=1.0
        )
    
    def supports_language(self, language: str) -> bool:
        return language in ["en", "zh"]
    
    def get_supported_formats(self) -> list:
        return [".wav", ".mp3"]
    
    def load_model(self) -> None:
        self._model = "mock_model"
    
    def unload_model(self) -> None:
        self._model = None


class TestTranscriptionEngine:
    """Test TranscriptionEngine base class."""
    
    def setup_method(self):
        """Setup for each test."""
        self.config = TranscriptionConfig()
        self.engine = MockTranscriptionEngine(self.config)
        
    def test_init(self):
        """Test engine initialization."""
        assert self.engine.config == self.config
        assert self.engine._model is None
        assert self.engine._device is None
        
    def test_is_model_loaded(self):
        """Test model loaded check."""
        assert self.engine.is_model_loaded() is False
        
        self.engine.load_model()
        assert self.engine.is_model_loaded() is True
        
        self.engine.unload_model()
        assert self.engine.is_model_loaded() is False
        
    def test_get_model_info(self):
        """Test model info retrieval."""
        info = self.engine.get_model_info()
        
        assert "name" in info
        assert "language" in info
        assert "device" in info
        assert "loaded" in info
        
        assert info["name"] == self.config.model_name
        assert info["loaded"] is False
        
    def test_validate_audio_file_success(self, tmp_path):
        """Test successful audio file validation."""
        audio_file = tmp_path / "test.wav"
        audio_file.touch()
        
        assert self.engine.validate_audio_file(audio_file) is True
        
    def test_validate_audio_file_not_found(self, tmp_path):
        """Test validation with missing file."""
        audio_file = tmp_path / "missing.wav"
        
        with pytest.raises(FileNotFoundError):
            self.engine.validate_audio_file(audio_file)
            
    def test_validate_audio_file_unsupported_format(self, tmp_path):
        """Test validation with unsupported format."""
        audio_file = tmp_path / "test.txt"
        audio_file.touch()
        
        with pytest.raises(UnsupportedAudioFormatError):
            self.engine.validate_audio_file(audio_file)


class TestEngineRegistry:
    """Test EngineRegistry class."""
    
    def setup_method(self):
        """Setup for each test."""
        self.registry = EngineRegistry()
        
    def test_register_engine(self):
        """Test engine registration."""
        self.registry.register("mock", MockTranscriptionEngine)
        
        assert "mock" in self.registry.list_engines()
        
    def test_register_invalid_engine(self):
        """Test registering invalid engine class."""
        class InvalidEngine:
            pass
            
        with pytest.raises(ValueError):
            self.registry.register("invalid", InvalidEngine)
            
    def test_get_engine(self):
        """Test getting registered engine."""
        self.registry.register("mock", MockTranscriptionEngine)
        
        engine_class = self.registry.get("mock")
        assert engine_class == MockTranscriptionEngine
        
    def test_get_unknown_engine(self):
        """Test getting unknown engine."""
        with pytest.raises(KeyError):
            self.registry.get("unknown")
            
    def test_create_engine(self):
        """Test creating engine instance."""
        self.registry.register("mock", MockTranscriptionEngine)
        
        config = TranscriptionConfig()
        engine = self.registry.create_engine("mock", config)
        
        assert isinstance(engine, MockTranscriptionEngine)
        assert engine.config == config


class TestWhisperEngine:
    """Test WhisperEngine class."""
    
    def setup_method(self):
        """Setup for each test."""
        self.config = TranscriptionConfig(
            model_name="base",
            language="en",
            device="cpu"
        )
        
    @patch('nice_tts.engines.transcription.whisper.whisper')
    @patch('nice_tts.engines.transcription.whisper.torch')
    def test_init_success(self, mock_torch, mock_whisper):
        """Test successful WhisperEngine initialization."""
        mock_torch.cuda.is_available.return_value = False
        
        engine = WhisperEngine(self.config)
        assert engine.config == self.config
        
    def test_init_invalid_model(self):
        """Test initialization with invalid model."""
        config = TranscriptionConfig(model_name="invalid_model")
        
        with pytest.raises(ValueError, match="Unsupported Whisper model"):
            WhisperEngine(config)
            
    def test_init_invalid_language(self):
        """Test initialization with invalid language."""
        config = TranscriptionConfig(language="invalid_lang")
        
        with pytest.raises(ValueError, match="Unsupported language"):
            WhisperEngine(config)
            
    def test_supports_language(self):
        """Test language support checking.""" 
        with patch('nice_tts.engines.transcription.whisper.torch'):
            engine = WhisperEngine(self.config)
            
        assert engine.supports_language("en") is True
        assert engine.supports_language("zh") is True
        assert engine.supports_language("invalid") is False
        
    def test_get_supported_formats(self):
        """Test supported formats retrieval."""
        with patch('nice_tts.engines.transcription.whisper.torch'):
            engine = WhisperEngine(self.config)
            
        formats = engine.get_supported_formats()
        assert ".wav" in formats
        assert ".mp3" in formats
        assert ".m4a" in formats
        
    @patch('nice_tts.engines.transcription.whisper.whisper')
    @patch('nice_tts.engines.transcription.whisper.torch')
    def test_load_model_cpu(self, mock_torch, mock_whisper):
        """Test model loading on CPU."""
        mock_torch.cuda.is_available.return_value = False
        
        mock_model = Mock()
        mock_whisper.load_model.return_value = mock_model
        
        engine = WhisperEngine(self.config)
        engine.load_model()
        
        assert engine._model == mock_model
        assert engine._device == "cpu"
        mock_whisper.load_model.assert_called_once()
        
    @patch('nice_tts.engines.transcription.whisper.whisper')
    @patch('nice_tts.engines.transcription.whisper.torch')
    def test_load_model_cuda(self, mock_torch, mock_whisper):
        """Test model loading on CUDA."""
        mock_torch.cuda.is_available.return_value = True
        
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_whisper.load_model.return_value = mock_model
        
        config = TranscriptionConfig(device="cuda")
        engine = WhisperEngine(config)
        engine.load_model()
        
        assert engine._device == "cuda"
        mock_model.to.assert_called_once_with("cuda")
        
    @patch('nice_tts.engines.transcription.whisper.whisper')
    @patch('nice_tts.engines.transcription.whisper.torch')
    def test_load_model_auto_detect(self, mock_torch, mock_whisper):
        """Test automatic device detection."""
        mock_torch.cuda.is_available.return_value = True
        
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_whisper.load_model.return_value = mock_model
        
        config = TranscriptionConfig(device="auto")
        engine = WhisperEngine(config)
        engine.load_model()
        
        assert engine._device == "cuda"
        
    @patch('nice_tts.engines.transcription.whisper.whisper')
    @patch('nice_tts.engines.transcription.whisper.torch')
    def test_transcribe_success(self, mock_torch, mock_whisper, tmp_path):
        """Test successful transcription."""
        mock_torch.cuda.is_available.return_value = False
        
        # Mock Whisper model
        mock_model = Mock()
        mock_result = {
            "text": "Hello world",
            "language": "en", 
            "segments": [],
            "duration": 2.5
        }
        mock_model.transcribe.return_value = mock_result
        mock_whisper.load_model.return_value = mock_model
        
        # Create test audio file
        audio_file = tmp_path / "test.wav"
        audio_file.touch()
        
        engine = WhisperEngine(self.config)
        engine.load_model()
        
        result = engine.transcribe(audio_file)
        
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.processing_time > 0
        
    @patch('nice_tts.engines.transcription.whisper.torch')
    def test_transcribe_model_not_loaded(self, mock_torch, tmp_path):
        """Test transcription with model not loaded."""
        mock_torch.cuda.is_available.return_value = False
        
        audio_file = tmp_path / "test.wav"
        audio_file.touch()
        
        engine = WhisperEngine(self.config)
        
        # Mock load_model to be called automatically
        with patch.object(engine, 'load_model') as mock_load:
            with patch.object(engine, '_model') as mock_model:
                mock_model.transcribe.return_value = {
                    "text": "test",
                    "language": "en",
                    "segments": []
                }
                
                result = engine.transcribe(audio_file)
                mock_load.assert_called_once()
                
    @patch('nice_tts.engines.transcription.whisper.torch')
    def test_check_gpu_availability(self, mock_torch):
        """Test GPU availability check."""
        mock_torch.__version__ = "2.0.0"
        mock_torch.cuda.is_available.return_value = True
        mock_torch.version.cuda = "11.8"
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3080"
        
        # Mock device properties
        mock_props = Mock()
        mock_props.total_memory = 10737418240  # 10GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.memory_reserved.return_value = 1073741824  # 1GB
        mock_torch.cuda.memory_allocated.return_value = 536870912  # 512MB
        
        info = WhisperEngine.check_gpu_availability()
        
        assert info["torch_version"] == "2.0.0"
        assert info["cuda_available"] is True
        assert info["cuda_version"] == "11.8"
        assert info["device_count"] == 1
        assert len(info["devices"]) == 1
        assert info["devices"][0]["name"] == "NVIDIA GeForce RTX 3080"


class TestWhisperEngineIntegration:
    """Integration tests for WhisperEngine."""
    
    @patch('nice_tts.engines.transcription.whisper.whisper')
    @patch('nice_tts.engines.transcription.whisper.torch')
    def test_full_transcription_workflow(self, mock_torch, mock_whisper, tmp_path):
        """Test complete transcription workflow."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        
        mock_model = Mock()
        mock_whisper.load_model.return_value = mock_model
        
        # Mock transcription result
        mock_result = {
            "text": "This is a test transcription.",
            "language": "en",
            "segments": [
                {
                    "text": "This is a test transcription.",
                    "avg_logprob": -0.2,
                    "tokens": ["This", "is", "a", "test", "transcription", "."]
                }
            ],
            "duration": 3.0
        }
        mock_model.transcribe.return_value = mock_result
        
        # Create test audio file
        audio_file = tmp_path / "test.wav"
        audio_file.touch()
        
        # Test workflow
        config = TranscriptionConfig(model_name="base", language="en", device="cpu")
        engine = WhisperEngine(config)
        
        # Validate file
        assert engine.validate_audio_file(audio_file) is True
        
        # Load model
        engine.load_model()
        assert engine.is_model_loaded() is True
        
        # Transcribe
        result = engine.transcribe(audio_file)
        
        # Verify result
        assert result.text == "This is a test transcription."
        assert result.language == "en"
        assert result.confidence > 0
        assert result.processing_time > 0
        assert result.metadata["model"] == "base"
        assert result.metadata["device"] == "cpu"
        
        # Unload model
        engine.unload_model()
        assert engine.is_model_loaded() is False


if __name__ == "__main__":
    pytest.main([__file__])