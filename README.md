# Nice-TTS: AI-Powered Transcription and Summarization (Chinese Optimized)

Nice-TTS is a powerful, batch-capable command-line tool that leverages AI to transcribe audio recordings, refine the transcriptions, and generate detailed meeting summaries. It is now optimized by default for **Chinese language** processing.

## Features

-   **AI-Powered Transcription**: Uses OpenAI's Whisper models to transcribe audio files. Defaults to Chinese.
-   **Chinese-Optimized LLM Processing**: Refinement and summarization prompts are written in Chinese to provide high-quality, context-aware results for Chinese language audio.
-   **Batch Processing**: Process a single audio file or all supported audio files in a directory.
-   **GPU Accelerated**: Automatically uses a CUDA-enabled GPU for transcription if available.
-   **Smart Processing**: Automatically skips completed steps if an output file is already present.
-   **Flexible Configuration**: Reads credentials from a local `.env` file or a global `~/.env` file.
-   **Organized Output**: Saves all generated files into a specified output directory.

## Requirements

-   Python 3.11 or higher.
-   `uv` for environment and package management (recommended).
-   `ffmpeg`: Whisper requires `ffmpeg` to be installed on your system.
-   For GPU acceleration, a CUDA-enabled NVIDIA GPU with the appropriate drivers.
-   Access to an OpenAI-compatible LLM API with an API key.

## Installation

### Basic Installation
1.  **Clone the repository:** `git clone <repository_url> && cd nice-tts`
2.  **Create environment:** `uv venv`
3.  **Activate environment:** 
    - Linux/Mac: `source .venv/bin/activate`
    - Windows: `.venv\Scripts\activate`

### GPU Installation (Recommended)

For optimal performance with GPU acceleration:

```bash
# Option 1: Use the GPU installation script (Recommended)
python install-gpu.py

# Option 2: Manual installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv sync
```

**Prerequisites for GPU support:**
- NVIDIA GPU with CUDA compute capability 7.0 or higher
- NVIDIA CUDA 12.6 drivers installed
- Sufficient GPU memory (4GB+ recommended for Whisper models)

### CPU-Only Installation

If you don't have a CUDA-compatible GPU:

```bash
uv sync
```

## GPU Support Check

To check if your system is ready for GPU acceleration, run:
```bash
nice-tts check-gpu
```

## Configuration

The tool needs API credentials for an LLM. It checks for a `.env` file in the current directory, then in your home directory (`~/.env`).

-   Copy the example file: `cp .env.example .env`
-   Edit the `.env` file with your details (API key, base URL, model name).

## Usage

The main command is `process`. By default, it assumes the audio is Chinese (`--language zh`) and will produce a Chinese refined transcript and summary.

### Processing Chinese Audio (Default)
```bash
# Process a single file
nice-tts process /path/to/chinese_meeting.wav

# Process a whole directory
nice-tts process /path/to/recordings_folder/ --output-dir chinese_results
```

### Processing English Audio

You can still process other languages by specifying the language code. Note that the LLM prompts are optimized for Chinese, so results in other languages may vary.

```bash
nice-tts process /path/to/english_meeting.wav --language en
```

For all options, run `nice-tts --help`.

---
*This project was created with the assistance of an AI software engineer.*
