import typer
from pathlib import Path
import sys

from . import transcription
from . import llm

app = typer.Typer(
    name="nice-tts",
    help="A CLI tool to transcribe, refine, and summarize audio files.",
    add_completion=False,
)

SUPPORTED_AUDIO_EXTENSIONS = [".wav", ".mp3", ".m4a", ".ogg"]

@app.command()
def process(
    input_path: Path = typer.Argument(
        ...,
        help="Path to a single audio file or a directory containing audio files.",
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    whisper_model: str = typer.Option(
        "large-v3-turbo",
        "--model",
        "-m",
        help="The Whisper model to use for transcription.",
    ),
    language: str = typer.Option(
        "zh",
        "--language",
        "-l",
        help="The language of the audio for transcription.",
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
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-processing of all steps.",
    ),
):
    """
    Process a single audio file or all audio files in a directory.
    """
    # Determine the list of audio files to process
    if input_path.is_dir():
        typer.secho(f"Scanning directory for audio files: {input_path}", fg=typer.colors.CYAN)
        audio_files = [
            f for f in input_path.iterdir() if f.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        ]
        if not audio_files:
            typer.secho("No supported audio files found in the directory.", fg=typer.colors.RED)
            raise typer.Exit()
    else:
        audio_files = [input_path]

    typer.secho(f"Found {len(audio_files)} audio file(s) to process.", fg=typer.colors.CYAN)
    output_dir.mkdir(parents=True, exist_ok=True)
    typer.secho(f"Output will be saved in: {output_dir}", fg=typer.colors.CYAN)

    # Loop through each audio file and process it
    for i, audio_file in enumerate(audio_files, 1):
        typer.secho(f"\n--- Processing file {i}/{len(audio_files)}: {audio_file.name} ---", fg=typer.colors.BRIGHT_BLUE)

        base_name = audio_file.stem
        srt_path = output_dir / f"{base_name}.srt"
        refined_path = output_dir / f"{base_name}.fine.txt"
        summary_path = output_dir / f"{base_name}.md"

        # --- Step 1: Transcription ---
        if force or not srt_path.exists():
            try:
                typer.secho("Step 1: Transcribing...", fg=typer.colors.BLUE)
                transcription.transcribe_audio(
                    audio_path=str(audio_file),
                    output_srt_path=str(srt_path),
                    model_name=whisper_model,
                    language=language,
                )
                typer.secho(f"✔ Transcription successful: {srt_path}", fg=typer.colors.GREEN)
            except Exception as e:
                typer.secho(f"Error during transcription: {e}", fg=typer.colors.RED, err=True)
                continue # Skip to the next file
        else:
            typer.secho(f"Step 1: Transcription skipped (file exists): {srt_path}", fg=typer.colors.YELLOW)

        # --- Step 2: Refinement ---
        if force or not refined_path.exists():
            if not srt_path.exists():
                typer.secho(f"Error: Cannot refine. SRT file missing: {srt_path}", fg=typer.colors.RED, err=True)
                continue
            try:
                typer.secho("Step 2: Refining transcript...", fg=typer.colors.BLUE)
                llm.refine_transcript(srt_path=str(srt_path), output_fine_path=str(refined_path))
                typer.secho(f"✔ Refinement successful: {refined_path}", fg=typer.colors.GREEN)
            except Exception as e:
                typer.secho(f"Error during refinement: {e}", fg=typer.colors.RED, err=True)
                continue
        else:
            typer.secho(f"Step 2: Refinement skipped (file exists): {refined_path}", fg=typer.colors.YELLOW)

        # --- Step 3: Summarization ---
        if force or not summary_path.exists():
            if not refined_path.exists():
                typer.secho(f"Error: Cannot summarize. Refined file missing: {refined_path}", fg=typer.colors.RED, err=True)
                continue
            try:
                typer.secho("Step 3: Generating summary...", fg=typer.colors.BLUE)
                llm.summarize_transcript(
                    refined_text_path=str(refined_path),
                    original_audio_path=str(audio_file),
                    output_md_path=str(summary_path),
                )
                typer.secho(f"✔ Summarization successful: {summary_path}", fg=typer.colors.GREEN)
            except Exception as e:
                typer.secho(f"Error during summarization: {e}", fg=typer.colors.RED, err=True)
                continue
        else:
            typer.secho(f"Step 3: Summarization skipped (file exists): {summary_path}", fg=typer.colors.YELLOW)

    typer.secho("\n🎉 Batch processing complete!", fg=typer.colors.BRIGHT_GREEN)

if __name__ == "__main__":
    app()
