#!/usr/bin/env python3
"""Simple test script to run GPU check"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from nice_tts.cli.commands import app

if __name__ == "__main__":
    # Run the check-gpu command
    app(["check-gpu"])