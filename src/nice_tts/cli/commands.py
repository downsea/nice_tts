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
from ..core.config import get_config, config_service
from ..core.pipeline import ProcessingPipeline
from ..core.exceptions import NiceTTSError, FatalError
from ..engines.transcription.whisper import WhisperEngine
from ..engines.llm.base import registry as llm_registry
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
    """Process audio files through transcription, refinement, and summarization.
    
    This command processes audio files through a three-stage pipeline:
    
    1. **Transcription**: Convert audio to text using Whisper
    2. **Refinement**: Clean and improve transcribed text using LLM
    3. **Summarization**: Generate structured meeting notes in Markdown
    
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
            """Handle progress updates."""
            if verbose:
                # Detailed progress in verbose mode
                typer.echo(
                    f"Progress: {progress.current}/{progress.total} "
                    f"({progress.percentage:.1f}%) - {progress.stage}"
                )
        
        # Process files
        with logger.performance_timer("batch_processing", input_path=str(input_path)):
            result = pipeline.process_batch(input_path, progress_callback)
        
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
    providers = llm_registry.list_engines()
    for provider in providers:
        if provider == "openai":
            typer.echo(f"  ✅ {provider} (fully supported)")
        else:
            typer.echo(f"  🚧 {provider} (experimental)")
    
    # Audio formats
    typer.secho("\nSupported Audio Formats:", fg=typer.colors.BLUE, bold=True)
    formats = ", ".join(WhisperEngine.SUPPORTED_FORMATS)
    typer.echo(f"  {formats}")


@app.command()
def config(
    action: str = typer.Argument(
        ...,
        help="Action to perform: show, validate, init"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--file", "-f",
        help="Configuration file path",
        callback=validate_config_file
    )
) -> None:
    """Configuration management commands.
    
    **Actions:**
    
    - **show**: Display current configuration
    - **validate**: Validate configuration without processing
    - **init**: Create a sample configuration file
    
    **Examples:**
    
        # Show current configuration
        nice-tts config show
        
        # Validate specific config file
        nice-tts config validate --file .env.production
        
        # Create sample configuration
        nice-tts config init
    """
    if action == "show":
        _show_config(config_file)
    elif action == "validate":
        _validate_config(config_file)
    elif action == "init":
        _init_config(config_file)
    else:
        typer.secho(
            f"❌ Unknown action: {action}. Use: show, validate, or init",
            fg=typer.colors.RED,
            err=True
        )
        raise typer.Exit(1)


def _display_startup_info(config, input_path: Path) -> None:
    """Display startup information."""
    typer.secho("🎙️  Nice-TTS Audio Processing Pipeline", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"Input: {input_path}")
    typer.echo(f"Output: {config.output.directory}")
    typer.echo(f"Transcription: {config.transcription.model_name} ({config.transcription.language})")
    typer.echo(f"LLM Provider: {config.llm.provider}")
    
    if config.parallel_jobs > 1:
        typer.echo(f"Parallel Jobs: {config.parallel_jobs}")
    
    if config.output.force_reprocess:
        typer.secho("🔄 Force reprocessing enabled", fg=typer.colors.YELLOW)
    
    typer.echo()


def _display_results(result) -> None:
    """Display processing results."""
    typer.echo("\n" + "="*50)
    typer.secho("📊 Processing Results", fg=typer.colors.CYAN, bold=True)
    typer.echo("="*50)
    
    typer.echo(f"Total Files: {result.total_files}")
    typer.secho(f"✅ Successful: {result.successful_files}", fg=typer.colors.GREEN)
    
    if result.failed_files > 0:
        typer.secho(f"❌ Failed: {result.failed_files}", fg=typer.colors.RED)
    
    typer.echo(f"⏱️  Total Time: {result.total_processing_time:.2f}s")
    
    if result.total_files > 0:
        avg_time = result.total_processing_time / result.total_files
        typer.echo(f"📈 Average Time per File: {avg_time:.2f}s")


def _show_config(config_file: Optional[Path]) -> None:
    """Show current configuration."""
    try:
        config = get_config(config_file)
        
        typer.secho("📋 Current Configuration\n", fg=typer.colors.CYAN, bold=True)
        
        # Transcription settings
        typer.secho("Transcription:", fg=typer.colors.BLUE, bold=True)
        typer.echo(f"  Model: {config.transcription.model_name}")
        typer.echo(f"  Language: {config.transcription.language}")
        typer.echo(f"  Device: {config.transcription.device}")
        
        # LLM settings
        typer.secho("\nLLM:", fg=typer.colors.BLUE, bold=True)
        typer.echo(f"  Provider: {config.llm.provider}")
        typer.echo(f"  Model: {config.llm.model_name or 'Not set'}")
        typer.echo(f"  Base URL: {config.llm.base_url or 'Not set'}")
        typer.echo(f"  Max Tokens: {config.llm.max_tokens}")
        
        # Output settings
        typer.secho("\nOutput:", fg=typer.colors.BLUE, bold=True)
        typer.echo(f"  Directory: {config.output.directory}")
        typer.echo(f"  Force Reprocess: {config.output.force_reprocess}")
        
        # Global settings
        typer.secho("\nGlobal:", fg=typer.colors.BLUE, bold=True)
        typer.echo(f"  Parallel Jobs: {config.parallel_jobs}")
        typer.echo(f"  Log Level: {config.logging.level}")
        
    except Exception as e:
        typer.secho(f"❌ Error loading configuration: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


def _validate_config(config_file: Optional[Path]) -> None:
    """Validate configuration."""
    try:
        config = get_config(config_file)
        config_service.validate_config(config)
        
        typer.secho("✅ Configuration is valid", fg=typer.colors.GREEN)
        
    except Exception as e:
        typer.secho(f"❌ Configuration validation failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


def _init_config(config_file: Optional[Path]) -> None:
    """Initialize sample configuration."""
    config_path = config_file or Path(".env")
    
    if config_path.exists():
        if not typer.confirm(f"Configuration file {config_path} already exists. Overwrite?"):
            typer.secho("Configuration initialization cancelled", fg=typer.colors.YELLOW)
            return
    
    sample_config = """# Nice-TTS Configuration File

# OpenAI/LLM Settings (Required)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-4-turbo-preview

# LLM Provider (openai, claude)
LLM_PROVIDER=openai

# Transcription Settings
WHISPER_MODEL=large-v3-turbo
WHISPER_LANGUAGE=zh
WHISPER_DEVICE=auto

# Output Settings
OUTPUT_DIR=output
FORCE_REPROCESS=false

# Logging Settings
LOG_LEVEL=INFO
# LOG_FILE=nice-tts.log

# Performance Settings
PARALLEL_JOBS=1
"""
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(sample_config)
        
        typer.secho(f"✅ Sample configuration created: {config_path}", fg=typer.colors.GREEN)
        typer.secho("📝 Edit the file and add your API keys before running", fg=typer.colors.BLUE)
        
    except Exception as e:
        typer.secho(f"❌ Failed to create configuration: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()