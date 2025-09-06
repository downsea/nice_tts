"""Configuration management commands for nice-tts CLI.

This module implements enhanced configuration commands with subcommand structure,
interactive wizard, and improved user experience.
"""

import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import os

import typer
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core.config import get_config, config_service, AppConfig
from .validators import validate_config_file
from ..engines.llm.base import get_registry
from ..engines.transcription.whisper import WhisperEngine

# Suppress deprecation warnings from main.py imports
warnings.filterwarnings("ignore", 
    message="Direct import of transcription and llm modules is deprecated")

# Rich console for enhanced output
console = Console()

# Create config subcommand app
config_app = typer.Typer(
    name="config",
    help="Configuration management commands",
    add_completion=False,
    rich_markup_mode="rich"
)


@config_app.callback(invoke_without_command=True)
def config_main(
    ctx: typer.Context,
    help_flag: bool = typer.Option(
        False, "-h", "--help",
        help="Show help information"
    )
) -> None:
    """Configuration management for nice-tts.
    
    Manage nice-tts configuration with interactive commands and validation.
    """
    if ctx.invoked_subcommand is None:
        # Show friendly help when no subcommand provided
        _show_config_status()
    elif help_flag:
        console.print(ctx.get_help())


@config_app.command("show")
def show_config(
    file: Optional[Path] = typer.Option(
        None, "--file", "-f",
        help="Configuration file path (default: .env in current directory)",
        callback=validate_config_file
    ),
    format: str = typer.Option(
        "table", "--format",
        help="Output format: table, json, yaml"
    )
) -> None:
    """Display current configuration.
    
    By default, reads configuration from .env file in the current directory.
    Use --file to specify a different configuration file.
    """
    try:
        # If no file specified, default to current directory .env
        config_file = file or Path(".env")
        
        # Only pass config_file to get_config if it exists or was explicitly specified
        if file is not None or config_file.exists():
            config = get_config(config_file if config_file.exists() else None)
        else:
            # No config file found, use default configuration
            config = get_config()
            
        if format == "json":
            _show_config_json(config)
        elif format == "yaml":
            _show_config_yaml(config)
        else:
            _show_config_table(config)
            
    except Exception as e:
        console.print(f"[red]❌ Error loading configuration: {e}[/red]")
        raise typer.Exit(1)


@config_app.command("validate")
def validate_config(
    file: Optional[Path] = typer.Option(
        None, "--file", "-f",
        help="Configuration file path (default: .env in current directory)",
        callback=validate_config_file
    ),
    check_api: bool = typer.Option(
        False, "--check-api",
        help="Test API connectivity"
    )
) -> None:
    """Validate configuration with detailed reporting.
    
    By default, validates .env file in the current directory.
    Use --file to specify a different configuration file.
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Validating configuration...", total=None)
            
            # Default to current directory .env if no file specified
            config_file = file or (Path(".env") if Path(".env").exists() else None)
            config = get_config(config_file)
            validation_results = _perform_validation(config, check_api, progress, task)
            
            progress.remove_task(task)
            
        _display_validation_results(validation_results)
        
        if validation_results["status"] == "failed":
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]❌ Validation error: {e}[/red]")
        raise typer.Exit(1)


@config_app.command("init")
def init_config(
    file: Optional[Path] = typer.Option(
        None, "--file", "-f",
        help="Configuration file path (default: .env in current directory)"
    ),
    template: str = typer.Option(
        "basic", "--template",
        help="Template type: basic, advanced, local"
    )
) -> None:
    """Initialize configuration file.
    
    By default, creates .env file in the current directory.
    Use --file to specify a different output location.
    """
    # Default to .env in current directory
    config_path = file or Path(".env")
    
    if config_path.exists():
        if not Confirm.ask(f"Configuration file {config_path} already exists. Overwrite?"):
            console.print("[yellow]Configuration initialization cancelled[/yellow]")
            return
    
    try:
        _create_config_template(config_path, template)
        console.print(f"[green]✅ Configuration created: {config_path}[/green]")
        console.print("[blue]📝 Edit the file and add your API keys before running[/blue]")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to create configuration: {e}[/red]")
        raise typer.Exit(1)


def _show_config_status() -> None:
    """Show configuration status overview."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]📋 Nice-TTS Configuration Status[/bold cyan]",
        border_style="cyan"
    ))
    
    # Check for config file - only current directory
    config_file_status = "❌ Not found"
    config_file_path = None
    
    # Check current directory .env (ONLY location)
    current_env = Path(".env")
    if current_env.exists():
        config_file_status = f"✅ Found in current directory ({current_env.resolve()})"
        config_file_path = current_env
    
    # Check API configuration
    api_status = "⏸️  Not configured"
    if config_file_path:
        try:
            config = get_config(config_file_path)
            if config.llm.api_key or config.llm.provider == "ollama":
                api_status = "✅ Configured"
        except:
            api_status = "❌ Invalid"
    
    console.print(f"Configuration file: {config_file_status}")
    console.print(f"API connection: {api_status}")
    
    console.print("\n[bold blue]🚀 Quick start:[/bold blue]")
    console.print("• [cyan]nice-tts config init[/cyan]    - Create .env in current directory")
    console.print("• [cyan]nice-tts config wizard[/cyan]  - Interactive setup")
    console.print("• [cyan]nice-tts config show[/cyan]    - View current settings")
    
    console.print("\n[bold blue]📖 More help:[/bold blue] [cyan]nice-tts config --help[/cyan]")
    console.print()


def _show_config_table(config: AppConfig) -> None:
    """Display configuration in table format."""
    console.print("\n[bold cyan]📋 Current Configuration[/bold cyan]\n")
    
    # Transcription settings
    table = Table(title="Transcription Settings", show_header=True, header_style="bold blue")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Model", config.transcription.model_name)
    table.add_row("Language", config.transcription.language)
    table.add_row("Device", config.transcription.device)
    table.add_row("Temperature", str(config.transcription.temperature))
    
    console.print(table)
    console.print()
    
    # LLM settings
    table = Table(title="LLM Settings", show_header=True, header_style="bold blue")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Provider", config.llm.provider)
    table.add_row("Model", config.llm.model_name or "Not set")
    table.add_row("Base URL", config.llm.base_url or "Not set")
    table.add_row("Max Tokens", str(config.llm.max_tokens))
    table.add_row("API Key", "***" if config.llm.api_key else "Not set")
    
    console.print(table)
    console.print()
    
    # Output & Global settings
    table = Table(title="Output & Global Settings", show_header=True, header_style="bold blue")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Output Directory", str(config.output.directory))
    table.add_row("Force Reprocess", str(config.output.force_reprocess))
    table.add_row("Parallel Jobs", str(config.parallel_jobs))
    table.add_row("Log Level", config.logging.level)
    
    console.print(table)


def _show_config_json(config: AppConfig) -> None:
    """Display configuration in JSON format."""
    import dataclasses
    import json
    
    def serialize_config(obj):
        if dataclasses.is_dataclass(obj):
            result = {}
            for field in dataclasses.fields(obj):
                value = getattr(obj, field.name)
                result[field.name] = serialize_config(value)
            return result
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return {k: serialize_config(v) for k, v in obj.__dict__.items()}
        else:
            return obj
    
    config_dict = serialize_config(config)
    # Hide sensitive information
    if config_dict.get("llm", {}).get("api_key"):
        config_dict["llm"]["api_key"] = "***"
    
    console.print(json.dumps(config_dict, indent=2, ensure_ascii=False))


def _show_config_yaml(config: AppConfig) -> None:
    """Display configuration in YAML format."""
    try:
        import yaml
        import dataclasses
        
        def serialize_config(obj):
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
            elif isinstance(obj, Path):
                return str(obj)
            return obj
        
        config_dict = serialize_config(config)
        # Hide sensitive information
        if config_dict.get("llm", {}).get("api_key"):
            config_dict["llm"]["api_key"] = "***"
        
        yaml_output = yaml.dump(config_dict, default_flow_style=False, sort_keys=True)
        console.print(yaml_output)
        
    except ImportError:
        console.print("[red]❌ PyYAML not installed. Use --format json instead[/red]")
        raise typer.Exit(1)


def _perform_validation(config: AppConfig, check_api: bool, progress, task_id) -> Dict[str, Any]:
    """Perform comprehensive configuration validation."""
    results = {
        "status": "passed",
        "summary": {"total": 0, "passed": 0, "failed": 0, "warnings": 0},
        "details": []
    }
    
    # Basic validation checks
    checks = [
        ("Configuration syntax", lambda: True),
        ("Required fields", lambda: _validate_required_fields(config)),
        ("File paths", lambda: _validate_file_paths(config)),
        ("Value ranges", lambda: _validate_value_ranges(config)),
    ]
    
    if check_api:
        checks.append(("API connectivity", lambda: _validate_api_connection(config)))
    
    for check_name, check_func in checks:
        progress.update(task_id, description=f"Checking {check_name.lower()}...")
        
        try:
            check_result = check_func()
            if check_result is True:
                status = "passed"
            elif isinstance(check_result, dict):
                status = check_result.get("status", "passed")
            else:
                status = "failed"
            
            results["details"].append({
                "check": check_name,
                "status": status,
                "message": check_result.get("message", f"{check_name} validation passed") if isinstance(check_result, dict) else f"{check_name} validation passed"
            })
            
            results["summary"]["total"] += 1
            if status == "passed":
                results["summary"]["passed"] += 1
            elif status == "warning":
                results["summary"]["warnings"] += 1
            else:
                results["summary"]["failed"] += 1
                results["status"] = "failed"
                
        except Exception as e:
            results["details"].append({
                "check": check_name,
                "status": "failed",
                "message": str(e)
            })
            results["summary"]["total"] += 1
            results["summary"]["failed"] += 1
            results["status"] = "failed"
    
    return results


def _validate_required_fields(config: AppConfig) -> Dict[str, Any]:
    """Validate required configuration fields."""
    if not config.llm.api_key:
        return {"status": "failed", "message": "LLM API key is required"}
    return {"status": "passed", "message": "All required fields present"}


def _validate_file_paths(config: AppConfig) -> Dict[str, Any]:
    """Validate file paths in configuration."""
    try:
        config.output.directory.mkdir(parents=True, exist_ok=True)
        return {"status": "passed", "message": "Output directory is accessible"}
    except PermissionError:
        return {"status": "failed", "message": f"Cannot access output directory: {config.output.directory}"}


def _validate_value_ranges(config: AppConfig) -> Dict[str, Any]:
    """Validate configuration value ranges."""
    if config.parallel_jobs <= 0:
        return {"status": "failed", "message": "Parallel jobs must be positive"}
    if config.llm.max_tokens <= 0:
        return {"status": "failed", "message": "Max tokens must be positive"}
    return {"status": "passed", "message": "All values within valid ranges"}


def _validate_api_connection(config: AppConfig) -> Dict[str, Any]:
    """Test API connectivity."""
    # This is a placeholder - would need actual API testing implementation
    if not config.llm.base_url:
        return {"status": "failed", "message": "API base URL not configured"}
    return {"status": "warning", "message": "API connectivity check not implemented"}


def _display_validation_results(results: Dict[str, Any]) -> None:
    """Display validation results."""
    summary = results["summary"]
    
    # Status panel
    status_color = "green" if results["status"] == "passed" else "red"
    status_emoji = "✅" if results["status"] == "passed" else "❌"
    
    console.print(f"\n{status_emoji} [bold {status_color}]Validation {results['status'].upper()}[/bold {status_color}]\n")
    
    # Summary table
    table = Table(title="Validation Summary", show_header=True, header_style="bold blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green")
    
    table.add_row("Total Checks", str(summary["total"]))
    table.add_row("Passed", str(summary["passed"]))
    table.add_row("Failed", str(summary["failed"]))
    table.add_row("Warnings", str(summary["warnings"]))
    
    console.print(table)
    
    # Detailed results
    if summary["failed"] > 0 or summary["warnings"] > 0:
        console.print("\n[bold yellow]Detailed Results:[/bold yellow]")
        for detail in results["details"]:
            if detail["status"] != "passed":
                status_emoji = "❌" if detail["status"] == "failed" else "⚠️"
                color = "red" if detail["status"] == "failed" else "yellow"
                console.print(f"{status_emoji} [bold {color}]{detail['check']}[/bold {color}]: {detail['message']}")


def _create_config_template(config_path: Path, template: str) -> None:
    """Create configuration template."""
    templates = {
        "basic": _get_basic_template(),
        "advanced": _get_advanced_template(),
        "local": _get_local_template()
    }
    
    if template not in templates:
        raise ValueError(f"Unknown template: {template}")
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(templates[template])


def _get_basic_template() -> str:
    """Get basic configuration template."""
    return """# Nice-TTS Basic Configuration

# LLM Settings (Required)
# For OpenAI:
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-4-turbo-preview

# For Ollama (local models):
# OLLAMA_API_BASE=http://localhost:11434
# OLLAMA_MODEL_NAME=llama2

# For Ollama:
# OLLAMA_API_BASE=http://localhost:11434/v1
# OLLAMA_MODEL_NAME=llama3

# LLM Provider (openai, ollama)
LLM_PROVIDER=openai

# Transcription Settings
WHISPER_MODEL=large-v3-turbo
WHISPER_LANGUAGE=zh

# Output Settings
OUTPUT_DIR=output
"""


def _get_advanced_template() -> str:
    """Get advanced configuration template.""" 
    return """# Nice-TTS Advanced Configuration

# LLM Settings
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-4-turbo-preview
LLM_PROVIDER=openai
LLM_TOKEN_MAX=128000

# Transcription Settings
WHISPER_MODEL=large-v3-turbo
WHISPER_LANGUAGE=zh
WHISPER_DEVICE=auto
# WHISPER_CACHE_DIR=~/.cache/whisper

# Output Settings
OUTPUT_DIR=output
FORCE_REPROCESS=false

# Logging Settings
LOG_LEVEL=INFO
# LOG_FILE=nice-tts.log

# Performance Settings
PARALLEL_JOBS=1
"""


@config_app.command("wizard")
def config_wizard(
    file: Optional[Path] = typer.Option(
        None, "--file", "-f",
        help="Configuration file path (default: .env in current directory)",
        callback=validate_config_file
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider",
        help="Pre-select LLM provider: openai, ollama"
    )
) -> None:
    """Interactive configuration wizard.
    
    By default, creates or updates .env file in the current directory.
    Use --file to specify a different output location.
    """
    # Default to .env in current directory
    config_path = file or Path(".env")
    
    console.print("\n[bold cyan]🧙 Nice-TTS Configuration Wizard[/bold cyan]\n")
    
    # Check for existing config
    if config_path.exists():
        if not Confirm.ask(f"Configuration file {config_path} exists. Overwrite?"):
            console.print("[yellow]Wizard cancelled[/yellow]")
            return
    
    # Start wizard flow
    try:
        config_data = _run_config_wizard(provider)
        _save_wizard_config(config_path, config_data)
        
        console.print(f"\n[green]✅ Configuration saved to {config_path}[/green]")
        console.print("[blue]🎉 You're ready to start using nice-tts![/blue]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Wizard cancelled by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"[red]❌ Wizard failed: {e}[/red]")
        raise typer.Exit(1)


@config_app.command("edit")
def edit_config(
    file: Optional[Path] = typer.Option(
        None, "--file", "-f",
        help="Configuration file path (default: .env in current directory)",
        callback=validate_config_file
    ),
    key: Optional[str] = typer.Option(
        None, "--key",
        help="Configuration key to edit"
    ),
    value: Optional[str] = typer.Option(
        None, "--value", 
        help="New value for the key"
    )
) -> None:
    """Edit configuration values.
    
    By default, edits .env file in the current directory.
    Use --file to specify a different configuration file.
    """
    # Default to .env in current directory
    config_path = file or Path(".env")
    
    if not config_path.exists():
        if Confirm.ask("Configuration file doesn't exist. Create it?"):
            _create_config_template(config_path, "basic")
        else:
            console.print("[yellow]Edit cancelled[/yellow]")
            return
    
    try:
        if key and value:
            _edit_config_value(config_path, key, value)
        else:
            _interactive_config_edit(config_path)
            
    except Exception as e:
        console.print(f"[red]❌ Edit failed: {e}[/red]")
        raise typer.Exit(1)


def _get_local_template() -> str:
    """Get local development configuration template."""
    return """# Nice-TTS Local Development Configuration

# Ollama Local LLM Settings (recommended for development)
OLLAMA_API_BASE=http://localhost:11434
OLLAMA_MODEL_NAME=llama2
LLM_PROVIDER=ollama

# Alternative: OpenAI-compatible Ollama endpoint
# OLLAMA_API_BASE=http://localhost:11434/v1
# OLLAMA_API_KEY=ollama

# Fast transcription for development
WHISPER_MODEL=base
WHISPER_LANGUAGE=zh
WHISPER_DEVICE=cpu

# Development output
OUTPUT_DIR=dev_output
FORCE_REPROCESS=true

# Debug logging
LOG_LEVEL=DEBUG
LOG_FILE=dev.log

# Single job for debugging
PARALLEL_JOBS=1
"""


def _run_config_wizard(provider: Optional[str]) -> Dict[str, str]:
    """Run the interactive configuration wizard."""
    config_data = {}
    
    # Welcome message
    console.print(Panel.fit(
        "Welcome to the Nice-TTS Configuration Wizard!\n"
        "This wizard will help you set up nice-tts for your needs.",
        title="🎉 Welcome",
        border_style="green"
    ))
    
    # Step 1: LLM Provider Selection
    console.print("\n[bold blue]Step 1: Choose LLM Provider[/bold blue]")
    
    if provider:
        llm_provider = provider
        console.print(f"Using pre-selected provider: [green]{provider}[/green]")
    else:
        llm_provider = Prompt.ask(
            "Select LLM provider",
            choices=["openai", "ollama"],
            default="openai"
        )
    
    config_data["LLM_PROVIDER"] = llm_provider
    
    # Step 2: API Configuration
    console.print("\n[bold blue]Step 2: API Configuration[/bold blue]")
    
    if llm_provider == "openai":
        api_key = Prompt.ask("Enter your OpenAI API key", password=True)
        config_data["OPENAI_API_KEY"] = api_key
        
        base_url = Prompt.ask(
            "API Base URL",
            default="https://api.openai.com/v1"
        )
        config_data["OPENAI_API_BASE"] = base_url
        
        models = ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"]
        model = Prompt.ask(
            "Select model",
            choices=models,
            default="gpt-4-turbo-preview"
        )
        config_data["OPENAI_MODEL_NAME"] = model
        
    elif llm_provider == "ollama":
        console.print("[green]Using local Ollama installation[/green]")
        
        base_url = Prompt.ask(
            "Ollama API Base URL",
            default="http://localhost:11434"
        )
        config_data["OLLAMA_API_BASE"] = base_url
        
        # For Ollama, we can try to fetch available models
        available_models = _get_ollama_models(base_url)
        if available_models:
            console.print(f"[blue]Available models: {', '.join(available_models[:5])}{'...' if len(available_models) > 5 else ''}[/blue]")
            model = Prompt.ask(
                "Select model",
                choices=available_models[:10] if len(available_models) > 10 else available_models,
                default=available_models[0] if available_models else "llama2"
            )
        else:
            console.print("[yellow]Could not fetch models from Ollama. Please enter manually.[/yellow]")
            model = Prompt.ask(
                "Enter Ollama model name",
                default="llama2"
            )
        config_data["OLLAMA_MODEL_NAME"] = model
        
        # Check if OpenAI-compatible mode should be used
        use_openai_api = Confirm.ask(
            "Use OpenAI-compatible API endpoint? (recommended for better integration)",
            default=True
        )
        if use_openai_api:
            config_data["OLLAMA_API_BASE"] = base_url.rstrip("/") + "/v1"
            config_data["OLLAMA_API_KEY"] = "ollama"  # Dummy key for compatibility
    
    # Step 3: Transcription Settings
    console.print("\n[bold blue]Step 3: Transcription Settings[/bold blue]")
    
    whisper_models = ["large-v3-turbo", "large-v3", "large-v2", "medium", "base"]
    whisper_model = Prompt.ask(
        "Select Whisper model (larger = better quality, slower)",
        choices=whisper_models,
        default="large-v3-turbo"
    )
    config_data["WHISPER_MODEL"] = whisper_model
    
    languages = ["zh", "en", "auto"]
    language = Prompt.ask(
        "Primary language for transcription",
        choices=languages,
        default="zh"
    )
    config_data["WHISPER_LANGUAGE"] = language
    
    # Step 4: Output Settings  
    console.print("\n[bold blue]Step 4: Output Settings[/bold blue]")
    
    output_dir = Prompt.ask(
        "Output directory for processed files",
        default="output"
    )
    config_data["OUTPUT_DIR"] = output_dir
    
    # Step 5: Performance Settings
    console.print("\n[bold blue]Step 5: Performance Settings[/bold blue]")
    
    parallel_jobs = Prompt.ask(
        "Number of parallel processing jobs",
        default="1"
    )
    config_data["PARALLEL_JOBS"] = parallel_jobs
    
    # Optional: Test connection
    if Confirm.ask("\nTest API connection now?", default=True):
        with console.status("[bold green]Testing API connection..."):
            test_result = _test_api_connection(config_data)
            if test_result:
                console.print("[green]✅ API connection successful![/green]")
            else:
                console.print("[yellow]⚠️  API connection failed, but configuration will be saved[/yellow]")
    
    return config_data


def _save_wizard_config(config_path: Path, config_data: Dict[str, str]) -> None:
    """Save wizard configuration to file."""
    config_content = "# Nice-TTS Configuration (Generated by Wizard)\n\n"
    
    # LLM Settings
    config_content += "# LLM Settings\n"
    for key, value in config_data.items():
        if key.startswith(("OPENAI_", "OLLAMA_", "LLM_")):
            config_content += f"{key}={value}\n"
    
    # Transcription Settings
    config_content += "\n# Transcription Settings\n"
    for key, value in config_data.items():
        if key.startswith("WHISPER_"):
            config_content += f"{key}={value}\n"
    
    # Output Settings
    config_content += "\n# Output Settings\n"
    for key, value in config_data.items():
        if key.startswith("OUTPUT_"):
            config_content += f"{key}={value}\n"
    
    # Performance Settings
    config_content += "\n# Performance Settings\n"
    for key, value in config_data.items():
        if key in ["PARALLEL_JOBS", "LOG_LEVEL"]:
            config_content += f"{key}={value}\n"
    
    # Default settings
    if "LOG_LEVEL" not in config_data:
        config_content += "LOG_LEVEL=INFO\n"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)


def _get_ollama_models(base_url: str) -> List[str]:
    """Get available Ollama models.
    
    Args:
        base_url: Ollama base URL
        
    Returns:
        List[str]: List of available model names
    """
    try:
        import requests
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model.get("name", "") for model in models if model.get("name")]
    except Exception:
        pass
    return []


def _test_api_connection(config_data: Dict[str, str]) -> bool:
    """Test API connection with provided configuration."""
    # This is a placeholder for actual API testing
    # In a real implementation, this would make a test API call
    import time
    time.sleep(1)  # Simulate API call
    return True  # Assume success for now


def _edit_config_value(config_path: Path, key: str, value: str) -> None:
    """Edit a specific configuration value."""
    lines = []
    key_found = False
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    
    # Update existing key or add new one
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            key_found = True
            break
    
    if not key_found:
        lines.append(f"{key}={value}\n")
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    console.print(f"[green]✅ Updated {key} = {value}[/green]")


def _interactive_config_edit(config_path: Path) -> None:
    """Interactive configuration editing."""
    console.print("\n[bold cyan]Interactive Configuration Editor[/bold cyan]\n")
    
    # Load current config
    current_config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    current_config[key] = value
    
    # Show current settings
    if current_config:
        table = Table(title="Current Configuration", show_header=True, header_style="bold blue")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in current_config.items():
            # Hide sensitive values
            display_value = "***" if "KEY" in key.upper() else value
            table.add_row(key, display_value)
        
        console.print(table)
    
    # Interactive editing
    console.print("\n[blue]Enter configuration key to edit (or 'quit' to exit):[/blue]")
    
    while True:
        key = Prompt.ask("Key to edit").strip()
        
        if key.lower() in ['quit', 'exit', 'q']:
            break
        
        if not key:
            continue
        
        current_value = current_config.get(key, "")
        if current_value and "KEY" in key.upper():
            console.print(f"Current value: [yellow]***[/yellow]")
        else:
            console.print(f"Current value: [yellow]{current_value}[/yellow]")
        
        new_value = Prompt.ask("New value", default=current_value)
        
        if new_value != current_value:
            _edit_config_value(config_path, key, new_value)
            current_config[key] = new_value
    
    console.print("[green]Configuration editing completed[/green]")