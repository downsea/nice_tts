"""Test configuration management functionality."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from nice_tts.core.config import (
    ConfigService, TranscriptionConfig, LLMConfig, 
    OutputConfig, LoggingConfig, AppConfig, get_config
)
from nice_tts.core.exceptions import ValidationError


class TestTranscriptionConfig:
    """Test TranscriptionConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TranscriptionConfig()
        assert config.model_name == "large-v3-turbo"
        assert config.language == "zh"
        assert config.device == "auto"
        assert config.temperature == 0.0
        
    def test_invalid_temperature(self):
        """Test temperature validation."""
        with pytest.raises(ValueError):
            TranscriptionConfig(temperature=-1.0)
        
        with pytest.raises(ValueError):
            TranscriptionConfig(temperature=2.0)


class TestLLMConfig:
    """Test LLMConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LLMConfig()
        assert config.provider == "openai"
        assert config.max_tokens == 128000
        assert config.temperature == 0.2
        
    def test_invalid_temperature(self):
        """Test temperature validation."""
        with pytest.raises(ValueError):
            LLMConfig(temperature=-1.0)
            
        with pytest.raises(ValueError):
            LLMConfig(temperature=3.0)
    
    def test_invalid_max_tokens(self):
        """Test max_tokens validation."""
        with pytest.raises(ValueError):
            LLMConfig(max_tokens=0)
            
        with pytest.raises(ValueError):
            LLMConfig(max_tokens=-1)


class TestOutputConfig:
    """Test OutputConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OutputConfig()
        assert config.directory == Path("output")
        assert config.force_reprocess is False
        assert config.save_intermediate is True
        
    def test_string_directory_conversion(self):
        """Test automatic string to Path conversion."""
        config = OutputConfig(directory="test_dir")
        assert isinstance(config.directory, Path)
        assert str(config.directory) == "test_dir"


class TestLoggingConfig:
    """Test LoggingConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.console_output is True
        assert config.structured_logging is False
        
    def test_level_validation(self):
        """Test log level validation."""
        config = LoggingConfig(level="debug")
        assert config.level == "DEBUG"
        
        with pytest.raises(ValueError):
            LoggingConfig(level="INVALID")


class TestAppConfig:
    """Test AppConfig class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = AppConfig()
        assert isinstance(config.transcription, TranscriptionConfig)
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.output, OutputConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert config.parallel_jobs == 1
        assert config.verbose is False
        
    def test_invalid_parallel_jobs(self):
        """Test parallel jobs validation."""
        with pytest.raises(ValueError):
            AppConfig(parallel_jobs=0)
            
        with pytest.raises(ValueError):
            AppConfig(parallel_jobs=-1)


class TestConfigService:
    """Test ConfigService class."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.service = ConfigService()
        
    @patch.dict(os.environ, {}, clear=True)
    def test_load_empty_config(self):
        """Test loading configuration with no environment variables."""
        config = self.service.load_config()
        assert isinstance(config, AppConfig)
        assert config.transcription.model_name == "large-v3-turbo"
        
    @patch.dict(os.environ, {
        "WHISPER_MODEL": "base",
        "WHISPER_LANGUAGE": "en",
        "OPENAI_API_KEY": "test_key",
        "OPENAI_API_BASE": "https://api.openai.com/v1",
        "OPENAI_MODEL_NAME": "gpt-4",
        "LLM_TOKEN_MAX": "100000"
    })
    def test_load_env_config(self):
        """Test loading configuration from environment variables."""
        config = self.service.load_config()
        
        assert config.transcription.model_name == "base"
        assert config.transcription.language == "en"
        assert config.llm.api_key == "test_key"
        assert config.llm.base_url == "https://api.openai.com/v1"
        assert config.llm.model_name == "gpt-4"
        assert config.llm.max_tokens == 100000
        
    def test_cli_overrides(self):
        """Test CLI parameter overrides."""
        cli_overrides = {
            "transcription": {
                "model_name": "tiny",
                "language": "fr"
            },
            "output": {
                "force_reprocess": True
            }
        }
        
        config = self.service.load_config(cli_overrides=cli_overrides)
        
        assert config.transcription.model_name == "tiny"
        assert config.transcription.language == "fr"
        assert config.output.force_reprocess is True
        
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test_key",
        "OPENAI_API_BASE": "https://api.openai.com/v1", 
        "OPENAI_MODEL_NAME": "gpt-4"
    })
    def test_validate_config_success(self):
        """Test successful configuration validation."""
        config = self.service.load_config()
        assert self.service.validate_config(config) is True
        
    def test_validate_config_missing_api_key(self):
        """Test configuration validation with missing API key."""
        config = AppConfig()
        config.llm.api_key = None
        
        with pytest.raises(ValueError, match="LLM API key is required"):
            self.service.validate_config(config)
            
    def test_merge_configs(self):
        """Test configuration dictionary merging."""
        base = {
            "transcription": {"model_name": "base", "language": "en"},
            "output": {"directory": "output"}
        }
        
        override = {
            "transcription": {"language": "zh"},
            "llm": {"provider": "openai"}
        }
        
        result = self.service._merge_configs(base, override)
        
        assert result["transcription"]["model_name"] == "base"
        assert result["transcription"]["language"] == "zh"
        assert result["output"]["directory"] == "output"
        assert result["llm"]["provider"] == "openai"


class TestGlobalConfigFunction:
    """Test global get_config function."""
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test_key",
        "OPENAI_API_BASE": "https://api.openai.com/v1",
        "OPENAI_MODEL_NAME": "gpt-4"
    })
    def test_get_config_function(self):
        """Test get_config convenience function."""
        config = get_config()
        assert isinstance(config, AppConfig)
        assert config.llm.api_key == "test_key"


# Integration tests
class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_config_file_loading(self, tmp_path):
        """Test loading configuration from file."""
        config_file = tmp_path / ".env"
        config_content = """
WHISPER_MODEL=small
WHISPER_LANGUAGE=en
OPENAI_API_KEY=file_key
OPENAI_API_BASE=https://api.example.com/v1
OPENAI_MODEL_NAME=gpt-3.5-turbo
OUTPUT_DIR=custom_output
LOG_LEVEL=DEBUG
"""
        config_file.write_text(config_content)
        
        service = ConfigService()
        config = service.load_config(config_file=config_file)
        
        assert config.transcription.model_name == "small"
        assert config.transcription.language == "en"
        assert config.llm.api_key == "file_key"
        assert config.llm.base_url == "https://api.example.com/v1"
        assert config.llm.model_name == "gpt-3.5-turbo"
        assert str(config.output.directory) == "custom_output"
        assert config.logging.level == "DEBUG"


if __name__ == "__main__":
    pytest.main([__file__])