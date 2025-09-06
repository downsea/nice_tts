"""Configuration management system for nice-tts.

This module provides centralized configuration management with support for:
- Environment variable loading from global and local .env files
- Configuration validation and type conversion
- Hierarchical configuration merging (defaults < global < local < CLI args)
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field, asdict

# Conditional imports for optional dependencies
try:
    from dotenv import load_dotenv, find_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    def load_dotenv(*args, **kwargs):
        pass
    def find_dotenv(*args, **kwargs):
        return None

# Conditional import for typer (only used for warnings)
try:
    import typer
except ImportError:
    typer = None


@dataclass
class TranscriptionConfig:
    """Configuration for transcription engines."""
    model_name: str = "large-v3-turbo"
    language: str = "zh"
    device: str = "auto"  # auto, cpu, cuda
    cache_dir: Optional[str] = None
    temperature: float = 0.0
    best_of: int = 5
    beam_size: int = 5
    format_output: bool = True  # Format transcript with each sentence on new line
    line_break_mode: str = "sentence"  # "sentence" or "segment"
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.temperature < 0 or self.temperature > 1:
            raise ValueError("Temperature must be between 0 and 1")
        if self.line_break_mode not in ["sentence", "segment"]:
            raise ValueError("line_break_mode must be 'sentence' or 'segment'")


@dataclass  
class LLMConfig:
    """Configuration for LLM engines."""
    provider: str = "openai"  # openai, ollama
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    max_tokens: int = 128000
    temperature: float = 0.2
    max_retries: int = 3
    timeout: int = 60
    enable_chunking: bool = True  # Enable intelligent text chunking
    chunk_overlap: int = 100  # Token overlap between chunks
    safety_margin: float = 0.8  # Use 80% of max_tokens for safety
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 < self.safety_margin <= 1.0:
            raise ValueError("safety_margin must be between 0 and 1")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")


@dataclass
class OutputConfig:
    """Configuration for output settings."""
    directory: Path = field(default_factory=lambda: Path("output"))
    force_reprocess: bool = False
    save_intermediate: bool = True
    format_timestamps: bool = False
    
    def __post_init__(self):
        """Ensure directory is a Path object."""
        if isinstance(self.directory, str):
            self.directory = Path(self.directory)


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    file_path: Optional[Path] = None
    console_output: bool = True
    structured_logging: bool = False
    
    def __post_init__(self):
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {self.level}. Must be one of {valid_levels}")
        self.level = self.level.upper()


@dataclass
class AppConfig:
    """Main application configuration container."""
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Global settings
    parallel_jobs: int = 1
    verbose: bool = False
    
    def __post_init__(self):
        """Validate global settings."""
        if self.parallel_jobs <= 0:
            raise ValueError("parallel_jobs must be positive")


class ConfigService:
    """Service for loading and managing configuration."""
    
    def __init__(self):
        self._config_cache: Optional[AppConfig] = None
        self._env_loaded = False
    
    def load_config(
        self, 
        config_file: Optional[Path] = None,
        cli_overrides: Optional[Dict[str, Any]] = None
    ) -> AppConfig:
        """Load configuration from multiple sources.
        
        Args:
            config_file: Optional path to a specific .env file
            cli_overrides: Dictionary of CLI parameter overrides
            
        Returns:
            AppConfig: Merged configuration object
        """
        if not self._env_loaded:
            self._load_env_files(config_file)
            self._env_loaded = True
        
        # Start with default config
        config_dict = self._get_default_config()
        
        # Merge environment variables
        env_config = self._load_env_config()
        config_dict = self._merge_configs(config_dict, env_config)
        
        # Apply CLI overrides
        if cli_overrides:
            config_dict = self._merge_configs(config_dict, cli_overrides)
        
        # Create and validate config objects
        try:
            config = AppConfig(
                transcription=TranscriptionConfig(**config_dict.get("transcription", {})),
                llm=LLMConfig(**config_dict.get("llm", {})),
                output=OutputConfig(**config_dict.get("output", {})),
                logging=LoggingConfig(**config_dict.get("logging", {})),
                parallel_jobs=config_dict.get("parallel_jobs", 1),
                verbose=config_dict.get("verbose", False)
            )
            
            self._config_cache = config
            return config
            
        except (TypeError, ValueError) as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    def validate_config(self, config: AppConfig) -> bool:
        """Validate configuration object.
        
        Args:
            config: Configuration to validate
            
        Returns:
            bool: True if valid, raises exception if invalid
        """
        # Validate LLM configuration based on provider
        if config.llm.provider == "ollama":
            # Ollama doesn't require API key, but needs model name
            if not config.llm.model_name:
                raise ValueError("Ollama model name is required")
            # Set default base URL if not provided
            if not config.llm.base_url:
                config.llm.base_url = "http://localhost:11434"
        else:
            # OpenAI requires API key
            if not config.llm.api_key:
                raise ValueError("LLM API key is required")
            if not config.llm.base_url:
                raise ValueError("LLM base URL is required") 
            if not config.llm.model_name:
                raise ValueError("LLM model name is required")
        
        # Validate output directory is writable
        try:
            config.output.directory.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise ValueError(f"Cannot create output directory: {config.output.directory}")
        
        return True
    
    def get_transcription_config(self) -> TranscriptionConfig:
        """Get transcription configuration."""
        if self._config_cache is None:
            self.load_config()
        return self._config_cache.transcription
    
    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        if self._config_cache is None:
            self.load_config()
        return self._config_cache.llm
    
    def _load_env_files(self, config_file: Optional[Path] = None) -> None:
        """Load environment variables from .env files.
        
        Load order (later overrides earlier):
        1. Local .env file (current working directory) - DEFAULT
        2. Specified config file (if provided)
        """
        # Load local .env file from current working directory (DEFAULT)
        local_dotenv_path = Path.cwd() / ".env"
        if local_dotenv_path.is_file():
            load_dotenv(dotenv_path=local_dotenv_path)
        
        # Load specified config file only if explicitly provided (overrides everything)
        if config_file and config_file.is_file():
            load_dotenv(dotenv_path=config_file, override=True)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "transcription": asdict(TranscriptionConfig()),
            "llm": asdict(LLMConfig()),
            "output": asdict(OutputConfig()),
            "logging": asdict(LoggingConfig()),
            "parallel_jobs": 1,
            "verbose": False
        }
    
    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {
            "transcription": {},
            "llm": {},
            "output": {},
            "logging": {}
        }
        
        # Transcription settings
        if model := os.getenv("WHISPER_MODEL"):
            config["transcription"]["model_name"] = model
        if language := os.getenv("WHISPER_LANGUAGE"):
            config["transcription"]["language"] = language
        if device := os.getenv("WHISPER_DEVICE"):
            config["transcription"]["device"] = device
        if cache_dir := os.getenv("WHISPER_CACHE_DIR"):
            config["transcription"]["cache_dir"] = cache_dir
        
        # New formatting options
        if format_output := os.getenv("WHISPER_FORMAT_OUTPUT"):
            config["transcription"]["format_output"] = format_output.lower() in ("true", "1", "yes")
        if line_break_mode := os.getenv("WHISPER_LINE_BREAK_MODE"):
            if line_break_mode in ["sentence", "segment"]:
                config["transcription"]["line_break_mode"] = line_break_mode
            elif typer:
                typer.secho(f"Warning: Invalid WHISPER_LINE_BREAK_MODE value: {line_break_mode}", fg=typer.colors.YELLOW)
        
        # LLM settings
        if api_key := os.getenv("OPENAI_API_KEY"):
            config["llm"]["api_key"] = api_key
        if base_url := os.getenv("OPENAI_API_BASE"):
            config["llm"]["base_url"] = base_url
        if model_name := os.getenv("OPENAI_MODEL_NAME"):
            config["llm"]["model_name"] = model_name
        if provider := os.getenv("LLM_PROVIDER"):
            config["llm"]["provider"] = provider
        
        # Support for Ollama-specific configuration
        if api_key := os.getenv("OLLAMA_API_KEY"):
            config["llm"]["api_key"] = api_key
        if base_url := os.getenv("OLLAMA_API_BASE"):
            config["llm"]["base_url"] = base_url
        if model_name := os.getenv("OLLAMA_MODEL_NAME"):
            config["llm"]["model_name"] = model_name
        if max_tokens := os.getenv("LLM_TOKEN_MAX"):
            try:
                config["llm"]["max_tokens"] = int(max_tokens)
            except ValueError:
                if typer:
                    typer.secho(f"Warning: Invalid LLM_TOKEN_MAX value: {max_tokens}", fg=typer.colors.YELLOW)
        
        # New LLM chunking options
        if enable_chunking := os.getenv("LLM_ENABLE_CHUNKING"):
            config["llm"]["enable_chunking"] = enable_chunking.lower() in ("true", "1", "yes")
        if chunk_overlap := os.getenv("LLM_CHUNK_OVERLAP"):
            try:
                config["llm"]["chunk_overlap"] = int(chunk_overlap)
            except ValueError:
                if typer:
                    typer.secho(f"Warning: Invalid LLM_CHUNK_OVERLAP value: {chunk_overlap}", fg=typer.colors.YELLOW)
        if safety_margin := os.getenv("LLM_SAFETY_MARGIN"):
            try:
                margin = float(safety_margin)
                if 0 < margin <= 1.0:
                    config["llm"]["safety_margin"] = margin
                elif typer:
                    typer.secho(f"Warning: LLM_SAFETY_MARGIN must be between 0 and 1: {safety_margin}", fg=typer.colors.YELLOW)
            except ValueError:
                if typer:
                    typer.secho(f"Warning: Invalid LLM_SAFETY_MARGIN value: {safety_margin}", fg=typer.colors.YELLOW)
        
        # Output settings  
        if output_dir := os.getenv("OUTPUT_DIR"):
            config["output"]["directory"] = Path(output_dir)
        if force := os.getenv("FORCE_REPROCESS"):
            config["output"]["force_reprocess"] = force.lower() in ("true", "1", "yes")
        
        # Logging settings
        if log_level := os.getenv("LOG_LEVEL"):
            config["logging"]["level"] = log_level
        if log_file := os.getenv("LOG_FILE"):
            config["logging"]["file_path"] = Path(log_file)
        
        # Global settings
        if parallel_jobs := os.getenv("PARALLEL_JOBS"):
            try:
                config["parallel_jobs"] = int(parallel_jobs)
            except ValueError:
                if typer:
                    typer.secho(f"Warning: Invalid PARALLEL_JOBS value: {parallel_jobs}", fg=typer.colors.YELLOW)
        
        return config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries recursively."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result


# Global configuration service instance
config_service = ConfigService()


def get_config(
    config_file: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None
) -> AppConfig:
    """Convenience function to get configuration."""
    return config_service.load_config(config_file, cli_overrides)