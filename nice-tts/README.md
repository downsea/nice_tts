# Nice-TTS: AI-Powered Transcription and Summarization

Nice-TTS is a powerful, batch-capable command-line tool that leverages AI to transcribe audio recordings, refine the transcriptions, and generate detailed meeting summaries. It's designed for efficient processing of single audio files or entire directories of recordings.

## Features

-   **AI-Powered Transcription**: Uses OpenAI's Whisper models to transcribe audio files.
-   **Batch Processing**: Process a single audio file or all supported audio files in a directory.
-   **Configurable Language**: Supports transcription for various languages (defaults to Chinese, `zh`).
-   **LLM-Based Refinement**: Connects to any OpenAI-compatible LLM to clean up the raw transcript.
-   **Intelligent Summarization**: Generates a comprehensive meeting summary in Markdown format.
-   **Smart Processing**: Automatically skips completed steps if an output file is already present, saving time and resources. Can be overridden with a `--force` flag.
-   **Flexible Configuration**: Reads credentials from a local `.env` file or a global `~/.env` file.
-   **Organized Output**: Saves all generated files into a specified output directory.

## Requirements

-   Python 3.11 or higher.
-   `uv` for environment and package management (recommended).
-   `ffmpeg`: Whisper requires `ffmpeg` to be installed on your system.
-   Access to an OpenAI-compatible LLM API with an API key.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd nice-tts
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -e .
    ```

## Configuration

The tool needs API credentials to connect to an LLM. It will look for a `.env` file first in the current directory, and if not found, it will check your home directory for a `~/.env` file.

1.  **Create a `.env` file** in your project directory or your home directory. You can copy the example file:
    ```bash
    cp .env.example .env
    ```

2.  **Edit the `.env` file** with your details.

    ```dotenv
    # .env - OpenAI-compatible LLM configuration
    OPENAI_API_KEY="your_api_key_here"
    OPENAI_API_BASE="https://api.openai.com/v1"
    OPENAI_MODEL_NAME="gpt-4"
    ```

## Usage

The main command is `process`, which runs the full pipeline on a given input path.

```
$ nice-tts --help
```
```
 Usage: nice-tts [OPTIONS] INPUT_PATH

 Process a single audio file or all audio files in a directory.

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    input_path      PATH  Path to a single audio file or a directory        │
│                            containing audio files.                           │
│                            [required]                                        │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --model       -m      TEXT       The Whisper model to use for transcription. │
│                                  [default: large-v3-turbo]                   │
│ --language    -l      TEXT       The language of the audio for transcription.│
│                                  [default: zh]                               │
│ --output-dir  -o      DIRECTORY  The directory to save output files.         │
│                                  [default: out]                              │
│ --force       -f                 Force re-processing of all steps.           │
│ --help                           Show this message and exit.                 │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Processing a Single File

To process a single audio file named `meeting.wav`:
```bash
nice-tts process /path/to/meeting.wav
```

### Processing a Directory

To process all supported audio files (`.wav`, `.mp3`, etc.) in a directory named `project_recordings`:
```bash
nice-tts process /path/to/project_recordings/ --output-dir project_results
```

## Output Files

The tool will generate up to three files for each processed audio file in the specified output directory (defaulting to `out/`):

1.  **SRT Transcript (`.srt`)**
2.  **Refined Text (`.fine.txt`)**
3.  **Markdown Summary (`.md`)**

---
*This project was created with the assistance of an AI software engineer.*
