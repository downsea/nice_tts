"""Logging management system for nice-tts.

This module provides a centralized logging system with support for:
- Console and file output
- Structured logging with JSON format
- Progress tracking and performance metrics
- Error aggregation and reporting
"""

import sys
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, IO
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum

import typer

from ..core.config import LoggingConfig


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"  
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ProgressInfo:
    """Progress tracking information."""
    
    current: int
    total: int
    stage: str
    file_name: Optional[str] = None
    start_time: Optional[float] = None
    
    @property
    def percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time since start."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    @property
    def eta(self) -> float:
        """Estimate time to completion."""
        if self.current == 0 or self.start_time is None or self.total == 0:
            return 0.0
        
        elapsed = self.elapsed_time
        # Avoid division by zero
        if elapsed == 0:
            return 0.0
            
        rate = self.current / elapsed
        remaining = self.total - self.current
        
        # Avoid division by zero
        if rate <= 0:
            return 0.0
            
        return remaining / rate if rate > 0 else 0.0


class LogFormatter(logging.Formatter):
    """Custom log formatter with color support for console output."""
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def __init__(self, use_colors: bool = True, structured: bool = False):
        """Initialize formatter.
        
        Args:
            use_colors: Enable color output
            structured: Use structured JSON format
        """
        self.use_colors = use_colors
        self.structured = structured
        
        if structured:
            super().__init__()
        else:
            fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            super().__init__(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record."""
        if self.structured:
            return self._format_structured(record)
        else:
            return self._format_colored(record)
    
    def _format_structured(self, record: logging.LogRecord) -> str:
        """Format as structured JSON log."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data, ensure_ascii=False)
    
    def _format_colored(self, record: logging.LogRecord) -> str:
        """Format with color support."""
        formatted = super().format(record)
        
        if self.use_colors and sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, '')
            reset = self.COLORS['RESET']
            return f"{color}{formatted}{reset}"
        
        return formatted


class Logger:
    """Centralized logger for nice-tts."""
    
    def __init__(self, config: LoggingConfig):
        """Initialize logger.
        
        Args:
            config: Logging configuration
        """
        self.config = config
        self._setup_logger()
        self._performance_data: List[Dict[str, Any]] = []
        self._error_count = 0
        
    def _setup_logger(self) -> None:
        """Setup logging configuration."""
        # Get root logger for nice-tts
        self.logger = logging.getLogger('nice_tts')
        self.logger.setLevel(getattr(logging, self.config.level))
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        if self.config.console_output:
            console_handler = logging.StreamHandler(sys.stderr)
            console_formatter = LogFormatter(
                use_colors=True,
                structured=self.config.structured_logging
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.file_path:
            try:
                # Ensure log directory exists
                self.config.file_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(
                    self.config.file_path,
                    mode='a',
                    encoding='utf-8'
                )
                file_formatter = LogFormatter(
                    use_colors=False,
                    structured=self.config.structured_logging
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
                
            except Exception as e:
                # Log to console if file logging fails
                self.logger.warning(f"Failed to setup file logging: {e}")
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._error_count += 1
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._error_count += 1
        self._log(LogLevel.CRITICAL, message, **kwargs)
    
    def _log(self, level: LogLevel, message: str, **kwargs) -> None:
        """Internal logging method."""
        # Create log record with extra fields
        extra_fields = {k: v for k, v in kwargs.items() if v is not None}
        
        # Get logger method
        log_method = getattr(self.logger, level.value.lower())
        
        if extra_fields and self.config.structured_logging:
            # For structured logging, add extra fields to record
            record = self.logger.makeRecord(
                name=self.logger.name,
                level=getattr(logging, level.value),
                fn="",
                lno=0,
                msg=message,
                args=(),
                exc_info=None
            )
            record.extra_fields = extra_fields
            self.logger.handle(record)
        else:
            # For regular logging, just log the message
            log_method(message)
    
    @contextmanager
    def performance_timer(self, operation: str, **context):
        """Context manager for timing operations.
        
        Args:
            operation: Name of the operation being timed
            **context: Additional context information
        """
        start_time = time.time()
        
        self.debug(f"Starting {operation}", operation=operation, **context)
        
        try:
            yield
            
            elapsed = time.time() - start_time
            
            # Record performance data
            perf_data = {
                "operation": operation,
                "elapsed_time": elapsed,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                **context
            }
            self._performance_data.append(perf_data)
            
            self.info(
                f"Completed {operation} in {elapsed:.2f}s",
                operation=operation,
                elapsed_time=elapsed,
                **context
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            
            # Record failure
            perf_data = {
                "operation": operation,
                "elapsed_time": elapsed,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e),
                **context
            }
            self._performance_data.append(perf_data)
            
            self.error(
                f"Failed {operation} after {elapsed:.2f}s: {e}",
                operation=operation,
                elapsed_time=elapsed,
                error=str(e),
                **context
            )
            
            raise
    
    def log_progress(self, progress: ProgressInfo) -> None:
        """Log progress information.
        
        Args:
            progress: Progress information
        """
        message = f"Progress: {progress.current}/{progress.total} ({progress.percentage:.1f}%)"
        
        if progress.file_name:
            message += f" - {progress.file_name}"
        
        if progress.stage:
            message += f" ({progress.stage})"
        
        # Add ETA if available
        if progress.eta > 0:
            eta_minutes = int(progress.eta // 60)
            eta_seconds = int(progress.eta % 60)
            message += f" - ETA: {eta_minutes:02d}:{eta_seconds:02d}"
        
        self.info(
            message,
            progress_current=progress.current,
            progress_total=progress.total,
            progress_percentage=progress.percentage,
            stage=progress.stage,
            file_name=progress.file_name,
            eta=progress.eta
        )
    
    def log_file_processing(
        self, 
        file_path: str, 
        stage: str, 
        status: str, 
        **context
    ) -> None:
        """Log file processing events.
        
        Args:
            file_path: Path to file being processed
            stage: Processing stage
            status: Processing status (started, completed, failed, skipped)
            **context: Additional context
        """
        file_name = Path(file_path).name
        message = f"File {status}: {file_name} ({stage})"
        
        log_level = LogLevel.INFO
        if status == "failed":
            log_level = LogLevel.ERROR
        elif status == "skipped":
            log_level = LogLevel.WARNING
        
        self._log(
            log_level,
            message,
            file_path=file_path,
            file_name=file_name,
            stage=stage,
            status=status,
            **context
        )
    
    def log_batch_summary(self, files_processed: int, files_total: int, elapsed_time: float) -> None:
        """Log batch processing summary.
        
        Args:
            files_processed: Number of files successfully processed
            files_total: Total number of files
            elapsed_time: Total elapsed time
        """
        success_rate = (files_processed / files_total) * 100 if files_total > 0 else 0
        
        self.info(
            f"Batch completed: {files_processed}/{files_total} files "
            f"({success_rate:.1f}% success) in {elapsed_time:.2f}s",
            files_processed=files_processed,
            files_total=files_total,
            success_rate=success_rate,
            elapsed_time=elapsed_time,
            errors_encountered=self._error_count
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics.
        
        Returns:
            Dict with performance statistics
        """
        if not self._performance_data:
            return {"total_operations": 0}
        
        successful_ops = [op for op in self._performance_data if op["status"] == "success"]
        failed_ops = [op for op in self._performance_data if op["status"] == "failed"]
        
        total_time = sum(op["elapsed_time"] for op in self._performance_data)
        avg_time = total_time / len(self._performance_data)
        
        # Group by operation type
        by_operation = {}
        for op in self._performance_data:
            op_name = op["operation"]
            if op_name not in by_operation:
                by_operation[op_name] = []
            by_operation[op_name].append(op["elapsed_time"])
        
        operation_stats = {}
        for op_name, times in by_operation.items():
            operation_stats[op_name] = {
                "count": len(times),
                "total_time": sum(times),
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times)
            }
        
        return {
            "total_operations": len(self._performance_data),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "success_rate": len(successful_ops) / len(self._performance_data) * 100,
            "total_time": total_time,
            "average_time": avg_time,
            "by_operation": operation_stats,
            "errors_encountered": self._error_count
        }


class ProgressReporter:
    """Progress reporter with visual indicators."""
    
    def __init__(self, logger: Logger, use_colors: bool = True):
        """Initialize progress reporter.
        
        Args:
            logger: Logger instance
            use_colors: Enable colored output
        """
        self.logger = logger
        self.use_colors = use_colors and sys.stderr.isatty()
        self._last_progress_line = ""
    
    def report_progress(self, progress: ProgressInfo, show_bar: bool = True) -> None:
        """Report progress with optional progress bar.
        
        Args:
            progress: Progress information
            show_bar: Show visual progress bar
        """
        # Log structured progress
        self.logger.log_progress(progress)
        
        # Show visual progress bar if requested
        if show_bar:
            self._show_progress_bar(progress)
    
    def _show_progress_bar(self, progress: ProgressInfo) -> None:
        """Show enhanced visual progress bar in console with ETA and stage info."""
        if not sys.stderr.isatty():
            return  # Don't show progress bar in non-interactive mode
        
        bar_width = 40
        filled = int(bar_width * progress.percentage / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Clear previous line
        if self._last_progress_line:
            sys.stderr.write("\r" + " " * len(self._last_progress_line) + "\r")
        
        # Create progress line with enhanced information
        line = f"Processing Files: [{bar}] {progress.percentage:5.1f}% ({progress.current}/{progress.total})"
        
        # Add current file information if available
        if progress.file_name:
            # Truncate long filenames
            display_name = progress.file_name
            if len(display_name) > 30:
                display_name = "..." + display_name[-27:]
            line += f"\nCurrent File: {display_name}"
        
        # Add stage information
        if progress.stage and progress.stage != "processing":
            line += f"\nStage: {progress.stage}"
            
            # Add ETA if available
            if progress.eta > 0:
                eta_minutes = int(progress.eta // 60)
                eta_seconds = int(progress.eta % 60)
                line += f" - ETA: {eta_minutes:02d}:{eta_seconds:02d}"
        
        # Add color if enabled
        if self.use_colors:
            if progress.percentage >= 100:
                color = "\033[32m"  # Green
            elif progress.percentage >= 75:
                color = "\033[33m"  # Yellow
            else:
                color = "\033[36m"  # Cyan
            
            line = f"{color}{line}\033[0m"
        
        sys.stderr.write(line)
        sys.stderr.flush()
        self._last_progress_line = line
    
    def clear_progress(self) -> None:
        """Clear progress bar from console."""
        if self._last_progress_line and sys.stderr.isatty():
            sys.stderr.write("\r" + " " * len(self._last_progress_line) + "\r")
            sys.stderr.flush()
            self._last_progress_line = ""


# Module-level convenience functions

_default_logger: Optional[Logger] = None

def setup_logging(config: LoggingConfig) -> Logger:
    """Setup global logger instance.
    
    Args:
        config: Logging configuration
        
    Returns:
        Logger instance
    """
    global _default_logger
    _default_logger = Logger(config)
    return _default_logger

def get_logger() -> Logger:
    """Get the default logger instance.
    
    Returns:
        Logger instance
        
    Raises:
        RuntimeError: If logger hasn't been setup
    """
    if _default_logger is None:
        raise RuntimeError("Logger not initialized. Call setup_logging() first.")
    return _default_logger