# Nice-TTS: AI-Powered Transcription and Summarization

Nice-TTS is a powerful command-line tool that leverages AI to transcribe audio recordings, refine the transcriptions, and generate detailed meeting summaries. It's designed to be a simple yet effective utility for anyone who needs to process voice recordings into well-structured text and summaries.

## Features

-   **AI-Powered Transcription**: Uses OpenAI's Whisper models to transcribe audio files (WAV, MP3, etc.) into text with high accuracy.
-   **LLM-Based Refinement**: Connects to any OpenAI-compatible Large Language Model (LLM) to clean up the raw transcript. It corrects punctuation, removes filler words, and improves readability.
-   **Intelligent Summarization**: Generates a comprehensive meeting summary in Markdown format, including an overview, discussion points, action items, and decisions made.
-   **Metadata Inference**: Smartly infers meeting details like date and topic from the filename and transcript content.
-   **Simple CLI**: Provides a user-friendly command-line interface for easy operation.

## Requirements

-   Python 3.11 or higher.
-   `uv` for environment and package management (recommended).
-   `ffmpeg`: Whisper requires `ffmpeg` to be installed on your system to process audio files. You can install it via your system's package manager (e.g., `sudo apt-get install ffmpeg` on Debian/Ubuntu, `brew install ffmpeg` on macOS).
-   Access to an OpenAI-compatible LLM API with an API key.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd nice-tts
    ```

2.  **Create a virtual environment and install dependencies using `uv`:**
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -e .
    ```
    *Note: The PyTorch dependencies for GPU support will be installed automatically based on the `pyproject.toml` configuration. For a specific CUDA version, you might need to adjust the installation command.*

## Configuration

Before you can use the refinement and summarization features, you need to provide your LLM API credentials.

1.  **Create a `.env` file** by copying the example file:
    ```bash
    cp .env.example .env
    ```

2.  **Edit the `.env` file** with your details.

    ```dotenv
    # .env - OpenAI-compatible LLM configuration
    # Your OpenAI API key or a key from a compatible service.
    OPENAI_API_KEY="your_api_key_here"

    # The base URL of the API. For OpenAI, this is https://api.openai.com/v1.
    # For other services (like local LLMs), change this to the correct endpoint.
    OPENAI_API_BASE="https://api.openai.com/v1"

    # The model to use for refinement and summarization.
    # e.g., "gpt-4", "gpt-3.5-turbo", or a custom model name.
    OPENAI_MODEL_NAME="gpt-4"
    ```

## Usage

The main command is `process`, which runs the full pipeline on a given audio file.

```
$ nice-tts --help
```
```
 Usage: nice-tts [OPTIONS] AUDIO_FILE

 Process an audio file through the full pipeline: Transcribe -> Refine ->
 Summarize.

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    audio_file      FILE  Path to the audio file to process (e.g., WAV,     │
│                            MP3).                                             │
│                            [required]                                        │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --model  -m      TEXT  The Whisper model to use for transcription (e.g.,     │
│                        tiny, base, small, medium, large).                    │
│                        [default: base]                                       │
│ --help                 Show this message and exit.                           │
╰──────────────────────────────────────────────────────────────────────────────╯
```

**Example:**

To process an audio file named `meeting-2023-10-27.wav` using the `small` Whisper model:
```bash
nice-tts process /path/to/meeting-2023-10-27.wav --model small
```

## Output Files

The tool will generate three files in the same directory as your input audio file:

1.  **SRT Transcript (`.srt`)**: A standard SubRip transcript file with timestamps.
    -   Example: `meeting-2023-10-27.srt`

2.  **Refined Text (`.fine.txt`)**: A clean, paragraph-formatted version of the transcript after being processed by the LLM.
    -   Example: `meeting-2023-10-27.fine.txt`

3.  **Markdown Summary (`.md`)**: A detailed meeting summary in Markdown format, with links to the other generated files.
    -   Example: `meeting-2023-10-27.md`

---
*This project was created with the assistance of an AI software engineer.*
