"""CLI parameter validators for nice-tts.

This module provides validation functions for command-line parameters
to ensure they are valid before processing begins.
"""

from pathlib import Path
from typing import Optional, List, Any, Dict

import typer

from ..core.config import get_config
from ..core.exceptions import ValidationError, ConfigurationError
from ..engines.transcription.whisper import WhisperEngine
from ..engines.llm.base import registry as llm_registry


def validate_input_path(path: Path) -> Path:
    """Validate input path exists and is accessible.
    
    Args:
        path: Input path to validate
        
    Returns:
        Path: Validated path
        
    Raises:
        typer.BadParameter: If path is invalid
    """
    if not path.exists():
        raise typer.BadParameter(f"Input path does not exist: {path}")
    
    if not (path.is_file() or path.is_dir()):
        raise typer.BadParameter(f"Input path must be a file or directory: {path}")
    
    if not os.access(path, os.R_OK):
        raise typer.BadParameter(f"Input path is not readable: {path}")
    
    return path


def validate_output_dir(path: Path) -> Path:
    """Validate output directory can be created and is writable.
    
    Args:
        path: Output directory path
        
    Returns:
        Path: Validated path
        
    Raises:
        typer.BadParameter: If path is invalid
    """
    try:
        # Try to create directory
        path.mkdir(parents=True, exist_ok=True)
        
        # Test write permissions
        test_file = path / ".test_write"
        test_file.touch()
        test_file.unlink()
        
        return path
        
    except PermissionError:
        raise typer.BadParameter(f"Cannot create or write to output directory: {path}")
    except Exception as e:
        raise typer.BadParameter(f"Invalid output directory: {e}")


def validate_whisper_model(model_name: str) -> str:
    """Validate Whisper model name.
    
    Args:
        model_name: Model name to validate
        
    Returns:
        str: Validated model name
        
    Raises:
        typer.BadParameter: If model is invalid
    """
    if model_name not in WhisperEngine.SUPPORTED_MODELS:
        available = ", ".join(WhisperEngine.SUPPORTED_MODELS)
        raise typer.BadParameter(
            f"Unsupported Whisper model: {model_name}. "
            f"Available models: {available}"
        )
    
    return model_name


def validate_language(language: str) -> str:
    """Validate language code.
    
    Args:
        language: Language code to validate
        
    Returns:
        str: Validated language code
        
    Raises:
        typer.BadParameter: If language is invalid
    """
    if language not in WhisperEngine.SUPPORTED_LANGUAGES:
        available = ", ".join(WhisperEngine.SUPPORTED_LANGUAGES[:10])  # Show first 10
        raise typer.BadParameter(
            f"Unsupported language: {language}. "
            f"Available languages include: {available} (and more)"
        )
    
    return language


def validate_llm_provider(provider: str) -> str:
    """Validate LLM provider.
    
    Args:
        provider: LLM provider name
        
    Returns:
        str: Validated provider name
        
    Raises:
        typer.BadParameter: If provider is invalid
    """
    available_providers = llm_registry.list_engines()
    
    if provider not in available_providers:
        available = ", ".join(available_providers)
        raise typer.BadParameter(
            f"Unsupported LLM provider: {provider}. "
            f"Available providers: {available}"
        )
    
    return provider


def validate_parallel_jobs(jobs: int) -> int:
    """Validate parallel jobs count.
    
    Args:
        jobs: Number of parallel jobs
        
    Returns:
        int: Validated job count
        
    Raises:
        typer.BadParameter: If job count is invalid
    """
    if jobs < 1:
        raise typer.BadParameter("Parallel jobs must be at least 1")
    
    if jobs > 16:
        raise typer.BadParameter("Parallel jobs cannot exceed 16 (to avoid resource exhaustion)")
    
    return jobs


def validate_config_file(path: Optional[Path]) -> Optional[Path]:
    """Validate configuration file.
    
    Args:
        path: Configuration file path
        
    Returns:
        Optional[Path]: Validated path or None
        
    Raises:
        typer.BadParameter: If config file is invalid
    """
    if path is None:
        return None
    
    if not path.exists():
        raise typer.BadParameter(f"Configuration file does not exist: {path}")
    
    if not path.is_file():
        raise typer.BadParameter(f"Configuration path is not a file: {path}")
    
    if path.suffix not in [".env", ".conf", ".config"]:
        raise typer.BadParameter(
            f"Configuration file should have .env, .conf, or .config extension: {path}"
        )
    
    return path


def validate_log_level(level: str) -> str:
    """Validate log level.
    
    Args:
        level: Log level to validate
        
    Returns:
        str: Validated log level
        
    Raises:
        typer.BadParameter: If log level is invalid
    """
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    level_upper = level.upper()
    
    if level_upper not in valid_levels:
        available = ", ".join(valid_levels)
        raise typer.BadParameter(
            f"Invalid log level: {level}. Valid levels: {available}"
        )
    
    return level_upper


def validate_configuration(config_overrides: Dict[str, Any]) -> None:
    """Validate complete configuration before processing.
    
    Args:
        config_overrides: Configuration overrides from CLI
        
    Raises:
        typer.BadParameter: If configuration is invalid
    """
    try:
        # Load and validate configuration
        config = get_config(cli_overrides=config_overrides)
        
        # Check required environment variables for LLM
        if not config.llm.api_key:
            raise typer.BadParameter(
                "LLM API key is required. Please set OPENAI_API_KEY in your .env file."
            )
        
        if not config.llm.base_url:
            raise typer.BadParameter(
                "LLM base URL is required. Please set OPENAI_API_BASE in your .env file."
            )
        
        if not config.llm.model_name:
            raise typer.BadParameter(
                "LLM model name is required. Please set OPENAI_MODEL_NAME in your .env file."
            )
        
    except (ValidationError, ConfigurationError) as e:
        raise typer.BadParameter(f"Configuration error: {e}")
    except Exception as e:
        raise typer.BadParameter(f"Unexpected configuration error: {e}")


# Import os for access check
import os