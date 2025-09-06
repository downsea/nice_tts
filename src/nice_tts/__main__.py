#!/usr/bin/env python3
"""Main entry point for Nice-TTS when run as a module."""

from .cli.commands import app

if __name__ == "__main__":
    app()