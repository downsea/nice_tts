import typer
from pathlib import Path
import sys

# By being in the same package, we can use relative imports.
from . import transcription
from . import llm

# Initialize the Typer app
app = typer.Typer(
    name="nice-tts",
    help="A CLI tool to transcribe, refine, and summarize audio files using AI.",
    add_completion=False,
)

@app.command()
def process(
    audio_file: Path = typer.Argument(
        ...,
        help="Path to the audio file to process (e.g., WAV, MP3).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    whisper_model: str = typer.Option(
        "large-v3",
        "--model",
        "-m",
        help="The Whisper model to use for transcription.",
    ),
    language: str = typer.Option(
        "zh",
        "--language",
        "-l",
        help="The language of the audio for transcription (e.g., 'en', 'zh').",
    ),
    output_dir: Path = typer.Option(
        "out",
        "--output-dir",
        "-o",
        help="The directory to save output files.",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
    token_max: int = typer.Option(
        128000,
        "--token-max",
        "-t",
        help="The maximum number of tokens to send to the LLM in a single request.",
    ),
):
    """
    Process an audio file through the full pipeline: Transcribe -> Refine -> Summarize.
    """
    typer.secho(f"🚀 Starting processing for: {audio_file.name}", fg=typer.colors.CYAN)

    # --- Step 1: Transcription ---
    txt_path = ""
    try:
        typer.secho("\nStep 1: Transcribing audio file...", fg=typer.colors.BLUE)
        # Assuming transcribe_audio is in the transcription module
        txt_path = transcription.transcribe_audio(str(audio_file), model_name=whisper_model)
        typer.secho(f"✔ Transcription successful. TXT file saved at: {txt_path}", fg=typer.colors.GREEN)
    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"An error occurred during transcription: {e}", fg=typer.colors.RED, err=True)
        typer.secho("This might be due to a missing dependency like `ffmpeg`. Please ensure it is installed and accessible in your system's PATH.", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)

    # --- Step 2: Refinement ---
    refined_path = ""
    try:
        typer.secho("\nStep 2: Refining transcript with LLM...", fg=typer.colors.BLUE)
        # Assuming refine_transcript is in the llm module
        refined_path = llm.refine_transcript(txt_path, token_max=token_max)
        typer.secho(f"✔ Refinement successful. Refined text saved at: {refined_path}", fg=typer.colors.GREEN)
    except FileNotFoundError as e:
        typer.secho(f"Error: Could not find the TXT file for refinement. {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"An error occurred during LLM refinement: {e}", fg=typer.colors.RED, err=True)
        typer.secho("Please ensure your .env file is correctly configured with your LLM provider's API key, base URL, and model name.", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)

    # --- Step 3: Summarization ---
    try:
        typer.secho("\nStep 3: Generating summary with LLM...", fg=typer.colors.BLUE)
        # Assuming summarize_transcript is in the llm module
        summary_path = llm.summarize_transcript(refined_path, str(audio_file), token_max=token_max)
        typer.secho(f"✔ Summarization successful. Summary saved at: {summary_path}", fg=typer.colors.GREEN)
    except FileNotFoundError as e:
        typer.secho(f"Error: Could not find the refined text file for summarization. {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"An error occurred during LLM summarization: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho("\n🎉 All steps completed successfully!", fg=typer.colors.BRIGHT_GREEN)
    typer.secho("You can find the generated files in the same directory as your audio file.", fg=typer.colors.WHITE)


if __name__ == "__main__":
    app()
