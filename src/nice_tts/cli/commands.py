"""Command handlers for the nice-tts CLI.

This module implements the command-line interface for nice-tts,
providing commands for audio processing, configuration management,
and system diagnostics.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import sys

import typer
import torch

from .validators import (
    validate_input_path, validate_output_dir, validate_whisper_model,
    validate_language, validate_llm_provider, validate_parallel_jobs,
    validate_config_file, validate_log_level, validate_configuration
)
from .config_commands import config_app
from ..core.config import get_config, config_service
from ..core.pipeline import ProcessingPipeline
from ..core.exceptions import NiceTTSError, FatalError
from ..engines.transcription.whisper import WhisperEngine
from ..engines.llm.base import get_registry
from ..utils.logger import setup_logging, ProgressInfo


# Create the main CLI application
app = typer.Typer(
    name="nice-tts",
    help="A CLI tool to transcribe, refine, and summarize audio files using AI.",
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command()
def process(
    input_path: Path = typer.Argument(
        ...,
        help="Path to audio file or directory containing audio files",
        callback=validate_input_path,
        metavar="INPUT"
    ),
    output_dir: Path = typer.Option(
        Path("output"),
        "--output-dir", "-o",
        help="Output directory for processed files",
        callback=validate_output_dir,
        metavar="DIR"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file (.env)",
        callback=validate_config_file,
        metavar="FILE"
    ),
    whisper_model: str = typer.Option(
        "large-v3-turbo",
        "--model", "-m",
        help="Whisper model to use for transcription",
        callback=validate_whisper_model,
        metavar="MODEL"
    ),
    language: str = typer.Option(
        "zh",
        "--language", "-l", 
        help="Language code for transcription (e.g., zh, en)",
        callback=validate_language,
        metavar="LANG"
    ),
    llm_provider: str = typer.Option(
        "openai",
        "--llm-provider",
        help="LLM provider for text processing",
        callback=validate_llm_provider,
        metavar="PROVIDER"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Force reprocessing of all files"
    ),
    parallel_jobs: int = typer.Option(
        1,
        "--parallel", "-j",
        help="Number of parallel processing jobs",
        callback=validate_parallel_jobs,
        metavar="N"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Set log level",
        callback=validate_log_level,
        metavar="LEVEL"
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        help="Save logs to file",
        metavar="FILE"
    )
) -> None:
    """Process audio files through transcription.
    
    This command processes audio files through transcription stage:
    
    1. **Transcription**: Convert audio to text using Whisper
    
    The tool automatically skips stages that have already been completed,
    unless --force is specified.
    
    **Examples:**
    
        # Process a single audio file
        nice-tts process meeting.wav
        
        # Process all audio files in a directory
        nice-tts process audio_files/ --output-dir results/
        
        # Force reprocessing with specific model
        nice-tts process meeting.wav --force --model large-v3
        
        # Process with parallel jobs
        nice-tts process audio_files/ --parallel 4
    """
    try:
        # Prepare configuration overrides
        config_overrides = {
            "transcription": {
                "model_name": whisper_model,
                "language": language
            },
            "llm": {
                "provider": llm_provider
            },
            "output": {
                "directory": output_dir,
                "force_reprocess": force
            },
            "logging": {
                "level": log_level,
                "file_path": log_file
            },
            "parallel_jobs": parallel_jobs,
            "verbose": verbose
        }
        
        # Validate configuration
        validate_configuration(config_overrides)
        
        # Load complete configuration
        config = get_config(config_file, config_overrides)
        
        # Setup logging
        logger = setup_logging(config.logging)
        
        # Display startup info
        _display_startup_info(config, input_path)
        
        # Create and run processing pipeline
        pipeline = ProcessingPipeline(config)
        
        def progress_callback(progress: ProgressInfo) -> None:
            """Handle progress updates with enhanced information."""
            if verbose:
                # Detailed progress in verbose mode
                eta_info = ""
                if progress.eta > 0:
                    eta_minutes = int(progress.eta // 60)
                    eta_seconds = int(progress.eta % 60)
                    eta_info = f" - ETA: {eta_minutes:02d}:{eta_seconds:02d}"
                
                typer.echo(
                    f"Progress: {progress.current}/{progress.total} "
                    f"({progress.percentage:.1f}%) - {progress.stage}{eta_info}"
                )
        
        # Process files
        with logger.performance_timer("batch_processing", input_path=str(input_path)):
            result = pipeline.process_batch(input_path, progress_callback)
        
        # Clear any remaining progress display
        pipeline.progress_reporter.clear_progress()
        
        # Display results
        _display_results(result)
        
        # Exit with error code if any files failed
        if result.failed_files > 0:
            typer.secho(
                f"\n⚠️  {result.failed_files} file(s) failed processing. Check logs for details.",
                fg=typer.colors.YELLOW,
                err=True
            )
            raise typer.Exit(1)
        
        typer.secho("\n🎉 All files processed successfully!", fg=typer.colors.GREEN)
        
    except typer.BadParameter as e:
        typer.secho(f"❌ Parameter error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    
    except FatalError as e:
        typer.secho(f"❌ Fatal error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    
    except NiceTTSError as e:
        typer.secho(f"❌ Processing error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    
    except KeyboardInterrupt:
        typer.secho("\n❌ Processing cancelled by user", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(130)
    
    except Exception as e:
        typer.secho(f"❌ Unexpected error: {e}", fg=typer.colors.RED, err=True)
        if verbose:
            import traceback
            typer.secho(traceback.format_exc(), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command()
def check_gpu() -> None:
    """Check GPU availability and display diagnostic information.
    
    This command checks for CUDA-enabled GPU availability and displays
    detailed information about the available hardware for transcription
    acceleration.
    """
    typer.secho("🔍 Checking GPU availability...\n", fg=typer.colors.CYAN, bold=True)
    
    try:
        gpu_info = WhisperEngine.check_gpu_availability()
        
        # Display PyTorch info
        typer.secho("PyTorch Information:", fg=typer.colors.BLUE, bold=True)
        typer.echo(f"  Version: {gpu_info['torch_version']}")
        
        # Display CUDA info
        if gpu_info["cuda_available"]:
            typer.secho("\n✅ CUDA GPU Available", fg=typer.colors.GREEN, bold=True)
            typer.echo(f"  CUDA Version: {gpu_info.get('cuda_version', 'Unknown')}")
            typer.echo(f"  Device Count: {gpu_info['device_count']}")
            
            # Display device details
            for device in gpu_info["devices"]:
                typer.secho(f"\n  Device {device['id']}: {device['name']}", fg=typer.colors.GREEN)
                
                memory_total = device["memory_total"] / (1024**3)  # Convert to GB
                memory_allocated = device["memory_allocated"] / (1024**3)
                memory_reserved = device["memory_reserved"] / (1024**3)
                
                typer.echo(f"    Total Memory: {memory_total:.2f} GB")
                typer.echo(f"    Allocated: {memory_allocated:.2f} GB")
                typer.echo(f"    Reserved: {memory_reserved:.2f} GB")
            
            typer.secho("\n💡 GPU acceleration will be used for transcription", fg=typer.colors.GREEN)
            
        else:
            typer.secho("\n⚠️  No CUDA GPU Available", fg=typer.colors.YELLOW, bold=True)
            typer.echo("  Transcription will use CPU (slower but still functional)")
            typer.secho("\n💡 To enable GPU acceleration:", fg=typer.colors.BLUE)
            typer.echo("  1. Install CUDA-compatible PyTorch")
            typer.echo("  2. Ensure NVIDIA drivers are installed")
            typer.echo("  3. Restart the application")
        
    except Exception as e:
        typer.secho(f"❌ Error checking GPU: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command()
def list_models() -> None:
    """List available transcription and LLM models.
    
    This command displays all available models and providers
    that can be used with nice-tts.
    """
    typer.secho("📋 Available Models and Providers\n", fg=typer.colors.CYAN, bold=True)
    
    # Whisper models
    typer.secho("Whisper Transcription Models:", fg=typer.colors.BLUE, bold=True)
    for model in WhisperEngine.SUPPORTED_MODELS:
        if "large" in model:
            typer.echo(f"  🔥 {model} (recommended for quality)")
        elif "turbo" in model:
            typer.echo(f"  ⚡ {model} (recommended for speed)")
        else:
            typer.echo(f"     {model}")
    
    # Supported languages
    typer.secho(f"\nSupported Languages ({len(WhisperEngine.SUPPORTED_LANGUAGES)}):", fg=typer.colors.BLUE, bold=True)
    lang_list = ", ".join(WhisperEngine.SUPPORTED_LANGUAGES[:20])
    typer.echo(f"  {lang_list}")
    if len(WhisperEngine.SUPPORTED_LANGUAGES) > 20:
        typer.echo(f"  ... and {len(WhisperEngine.SUPPORTED_LANGUAGES) - 20} more")
    
    # LLM providers
    typer.secho("\nLLM Providers:", fg=typer.colors.BLUE, bold=True)
    providers = get_registry().list_engines()
    for provider in providers:
        if provider == "openai":
            typer.echo(f"  ✅ {provider} (fully supported)")
        elif provider == "ollama":
            typer.echo(f"  🦙 {provider} (local models)")
        else:
            typer.echo(f"  🚧 {provider} (experimental)")
    
    # Audio formats
    typer.secho("\nSupported Audio Formats:", fg=typer.colors.BLUE, bold=True)
    formats = ", ".join(WhisperEngine.SUPPORTED_FORMATS)
    typer.echo(f"  {formats}")


# Add config subcommand
app.add_typer(config_app, name="config")


def _display_startup_info(config, input_path: Path) -> None:
    """Display enhanced startup information with better visual appeal."""
    # Enhanced startup display with visual separators and better formatting
    typer.secho("🎙️  Nice-TTS Audio Processing Pipeline", fg=typer.colors.CYAN, bold=True)
    typer.secho("═══════════════════════════════════════════════════════════════════════════════", fg=typer.colors.CYAN)
    
    # Input/Output information
    typer.echo(f"📁 Input:     {input_path}")
    typer.echo(f"📂 Output:    {config.output.directory}")
    
    # Transcription information
    typer.echo(f"🗣️  Transcription: {config.transcription.model_name} ({config.transcription.language}) on {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Parallel processing information
    if config.parallel_jobs > 1:
        typer.echo(f"⚡ Parallel Jobs: {config.parallel_jobs}")
    else:
        typer.echo("⚡ Parallel Jobs: 1 (sequential)")
    
    # Force reprocessing information
    if config.output.force_reprocess:
        typer.secho("🔄 Force Reprocess: Yes", fg=typer.colors.YELLOW)
    else:
        typer.echo("🔄 Force Reprocess: No")
    
    # Environment configuration section
    typer.echo("")
    typer.secho("[Environment Configuration]", fg=typer.colors.BLUE)
    
    # API configuration (masked for security)
    if hasattr(config.llm, 'api_key') and config.llm.api_key:
        masked_key = config.llm.api_key[-8:] if len(config.llm.api_key) > 8 else config.llm.api_key
        typer.echo(f"🔑 API Key:   ****************{masked_key}")
    
    if hasattr(config.llm, 'api_base') and config.llm.api_base:
        typer.echo(f"🌐 API Base:  {config.llm.api_base}")
    
    if hasattr(config.llm, 'model') and config.llm.model:
        typer.echo(f"🤖 Model:     {config.llm.model}")
    
    typer.echo()


def _display_results(result) -> None:
    """Display enhanced processing results with better formatting and detailed error information."""
    # Enhanced results summary with visual separators
    typer.echo()
    typer.secho("═══════════════════════════════════════════════════════════════════════════════", fg=typer.colors.CYAN)
    typer.secho("📊 Processing Results", fg=typer.colors.CYAN, bold=True)
    typer.secho("═══════════════════════════════════════════════════════════════════════════════", fg=typer.colors.CYAN)
    
    # Statistics - protect against division by zero
    success_rate = (result.successful_files / result.total_files * 100) if result.total_files > 0 else 0
    
    typer.echo(f"✅ Successful: {result.successful_files} files ({success_rate:.1f}%)")
    if result.failed_files > 0:
        typer.secho(f"❌ Failed:     {result.failed_files} files ({100-success_rate:.1f}%)", fg=typer.colors.RED)
    typer.echo(f"⏱️  Total Time: {int(result.total_processing_time // 60)}m {result.total_processing_time % 60:.1f}s")
    
    if result.total_files > 0:
        avg_time = result.total_processing_time / result.total_files
        typer.echo(f"📈 Average Time per File: {avg_time:.1f}s")
    
    # Detailed breakdown sections
    if result.successful_files > 0:
        typer.echo()
        typer.secho("[Successes]", fg=typer.colors.GREEN, bold=True)
        successful_results = [r for r in result.files_processed if r.success]
        for res in successful_results:
            # Display file and output path
            typer.echo(f"📁 {res.file_info.audio_path.name} → {res.stages_completed[-1].output_path if res.stages_completed and res.stages_completed[-1].output_path else 'N/A'}")
            
            # Display stage timing information
            stage_info = []
            for stage_result in res.stages_completed:
                stage_info.append(f"{stage_result.stage.value}: {stage_result.processing_time:.1f}s")
            
            if stage_info:
                typer.echo(f"   🕐 {', '.join(stage_info)}")
    
    if result.failed_files > 0:
        typer.echo()
        typer.secho("[Failures]", fg=typer.colors.RED, bold=True)
        failed_results = [r for r in result.files_processed if not r.success]
        for res in failed_results:
            typer.echo(f"📁 {res.file_info.audio_path.name}")
            
            if res.error:
                # Try to get more specific error information
                error_msg = str(res.error)
                error_type = type(res.error).__name__
                
                # Determine stage and details from error
                stage = "Unknown"
                details = ""
                
                if hasattr(res.error, 'stage'):
                    stage = getattr(res.error, 'stage', 'Unknown')
                elif res.stages_completed:
                    # Get the last stage that was attempted
                    last_stage = next((s for s in reversed(res.stages_completed) if not s.success), None)
                    if last_stage:
                        stage = last_stage.stage.value
                
                # Get error details
                if hasattr(res.error, 'details'):
                    error_details = getattr(res.error, 'details', {})
                    if error_details:
                        details = ", ".join([f"{k}: {v}" for k, v in error_details.items()])
                
                typer.secho(f"   Error: {error_type} - {error_msg}", fg=typer.colors.RED)
                typer.secho(f"   Stage: {stage}", fg=typer.colors.RED)
                if details:
                    typer.secho(f"   Details: {details}", fg=typer.colors.RED)
            typer.echo()
if __name__ == "__main__":
    app()