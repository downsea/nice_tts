"""Nice-TTS: AI-powered audio transcription and meeting summarization tool.

Nice-TTS is a command-line tool that provides high-quality audio transcription
and meeting summarization, optimized for Chinese language processing.

Main features:
- High-accuracy transcription using OpenAI Whisper
- LLM-powered text refinement and summarization
- Support for multiple audio formats
- GPU acceleration support
- Batch processing capabilities

Example usage:
    from nice_tts.core.pipeline import ProcessingPipeline
    from nice_tts.core.config import get_config
    
    config = get_config()
    pipeline = ProcessingPipeline(config)
    result = pipeline.process_batch(Path("audio_files/"))
"""

# Version info
__version__ = "0.2.0"
__author__ = "Jules"
__description__ = "AI-powered audio transcription and meeting summarization tool"

# Core exports (CLI app is imported lazily to avoid dependency issues)
from .core.config import get_config, AppConfig
from .core.exceptions import NiceTTSError

# Lazy import for CLI app to avoid typer dependency during core module imports
def get_app():
    """Get the CLI application (lazy import)."""
    from .cli.commands import app
    return app

# Re-export main components for convenience
__all__ = [
    "get_config", 
    "AppConfig",
    "NiceTTSError",
    "get_app",
    "__version__",
]