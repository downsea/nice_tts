"""Main entry point for the nice-tts CLI application.

This module provides the main entry point for the nice-tts application,
importing and exposing the CLI commands from the refactored command handlers.
"""

import warnings

# Suppress deprecation warnings to improve user experience
warnings.filterwarnings("ignore", 
    message="Direct import of transcription and llm modules is deprecated")

# Import the main CLI application from the new structure
from .cli.commands import app

# Re-export the main app for the entry point
__all__ = ["app"]

if __name__ == "__main__":
    app()
