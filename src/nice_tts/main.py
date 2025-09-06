"""Main entry point for the nice-tts CLI application.

This module provides the main entry point for the nice-tts application,
importing and exposing the CLI commands from the refactored command handlers.
"""

# Import the main CLI application from the new structure
from .cli.commands import app

# For backward compatibility, also import the old modules but mark them as deprecated
# This allows existing code to continue working while we transition
try:
    from . import transcription
    from . import llm
    import warnings
    warnings.warn(
        "Direct import of transcription and llm modules is deprecated. "
        "Use the new CLI interface instead.",
        DeprecationWarning,
        stacklevel=2
    )
except ImportError:
    # If the old modules don't exist, that's fine
    pass

# Re-export the main app for the entry point
__all__ = ["app"]

if __name__ == "__main__":
    app()
