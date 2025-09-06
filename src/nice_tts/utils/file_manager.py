"""File management utilities for nice-tts.

This module provides utilities for managing files throughout the processing pipeline,
including path resolution, file discovery, output management, and skip logic.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass
from enum import Enum

from ..core.config import OutputConfig
from ..core.exceptions import (
    FileNotFoundError,
    FileReadError,
    FileWriteError,
    ValidationError
)

logger = logging.getLogger('nice_tts')


class ProcessingStage(Enum):
    """Enumeration of processing stages."""
    TRANSCRIPTION = "transcription"


@dataclass
class FileInfo:
    """Information about a processing file."""
    
    audio_path: Path
    output_dir: Path
    base_name: str
    
    # Output file paths
    transcript_path: Path
    refined_path: Path
    summary_path: Path
    
    # Status tracking
    existing_stages: Set[ProcessingStage]
    
    @classmethod
    def from_audio_path(cls, audio_path: Path, output_dir: Path) -> "FileInfo":
        """Create FileInfo from audio file path.
        
        Args:
            audio_path: Path to audio file
            output_dir: Output directory
            
        Returns:
            FileInfo instance
        """
        base_name = audio_path.stem
        
        transcript_path = output_dir / f"{base_name}.txt"
        refined_path = output_dir / f"{base_name}.fine.txt"
        summary_path = output_dir / f"{base_name}.md"
        
        # Check which stages already exist
        existing_stages = set()
        if transcript_path.exists() and transcript_path.stat().st_size > 0:
            existing_stages.add(ProcessingStage.TRANSCRIPTION)
        
        return cls(
            audio_path=audio_path,
            output_dir=output_dir,
            base_name=base_name,
            transcript_path=transcript_path,
            refined_path=refined_path,
            summary_path=summary_path,
            existing_stages=existing_stages
        )
    
    def needs_processing(self, stage: ProcessingStage, force: bool = False) -> bool:
        """Check if a stage needs processing.
        
        Args:
            stage: Processing stage to check
            force: Force reprocessing even if output exists
            
        Returns:
            bool: True if processing is needed
        """
        if force:
            return True
        
        return stage not in self.existing_stages
    
    def get_stage_path(self, stage: ProcessingStage) -> Path:
        """Get output path for a processing stage.
        
        Args:
            stage: Processing stage
            
        Returns:
            Path: Output file path for the stage
        """
        if stage == ProcessingStage.TRANSCRIPTION:
            return self.transcript_path
        else:
            raise ValueError(f"Unknown processing stage: {stage}")
    
    def get_input_for_stage(self, stage: ProcessingStage) -> Optional[Path]:
        """Get input file path for a processing stage.
        
        Args:
            stage: Processing stage
            
        Returns:
            Path: Input file path, or None if not available
        """
        if stage == ProcessingStage.TRANSCRIPTION:
            return self.audio_path
        else:
            raise ValueError(f"Unknown processing stage: {stage}")


class FileManager:
    """Manager for file operations and path resolution."""
    
    # Supported audio formats
    SUPPORTED_AUDIO_EXTENSIONS = [".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"]
    
    def __init__(self, config: OutputConfig):
        """Initialize file manager.
        
        Args:
            config: Output configuration
        """
        self.config = config
        self.output_dir = config.directory
        
        # Ensure output directory exists
        self.ensure_output_directory()
    
    def discover_audio_files(self, input_path: Path) -> List[Path]:
        """Discover audio files from input path.
        
        Args:
            input_path: Path to file or directory
            
        Returns:
            List[Path]: List of audio file paths
            
        Raises:
            FileNotFoundError: If input path doesn't exist
            ValidationError: If no audio files found
        """
        if not input_path.exists():
            raise FileNotFoundError(
                f"Input path not found: {input_path}",
                file_path=str(input_path)
            )
        
        audio_files = []
        
        if input_path.is_file():
            # Single file
            if self.is_supported_audio_format(input_path):
                audio_files = [input_path]
            else:
                raise ValidationError(
                    f"Unsupported audio format: {input_path.suffix}",
                    details={"supported_formats": self.SUPPORTED_AUDIO_EXTENSIONS}
                )
        
        elif input_path.is_dir():
            # Directory - find all audio files
            for file_path in input_path.iterdir():
                if file_path.is_file() and self.is_supported_audio_format(file_path):
                    audio_files.append(file_path)
            
            # Sort for consistent processing order
            audio_files.sort()
        
        else:
            raise ValidationError(f"Input path is neither file nor directory: {input_path}")
        
        if not audio_files:
            raise ValidationError(
                f"No supported audio files found in: {input_path}",
                details={"supported_formats": self.SUPPORTED_AUDIO_EXTENSIONS}
            )
        
        # Log discovered files
        logger.info(f"Discovered {len(audio_files)} audio files")
        for audio_file in audio_files:
            logger.debug(f"Found audio file: {audio_file}")
        
        return audio_files
    
    def create_file_info(self, audio_path: Path) -> FileInfo:
        """Create FileInfo for an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            FileInfo: File information object
        """
        return FileInfo.from_audio_path(audio_path, self.output_dir)
    
    def get_processing_plan(
        self, 
        audio_files: List[Path], 
        force: bool = False
    ) -> List[Dict[str, Any]]:
        """Create a processing plan for audio files.
        
        Args:
            audio_files: List of audio file paths
            force: Force reprocessing of all stages
            
        Returns:
            List[Dict]: Processing plan with file info and required stages
        """
        plan = []
        
        # Handle empty audio files list gracefully
        if not audio_files:
            logger.warning("No audio files to process")
            return plan
        
        for audio_path in audio_files:
            file_info = self.create_file_info(audio_path)
            
            # Determine required stages
            required_stages = []
            for stage in ProcessingStage:
                if file_info.needs_processing(stage, force):
                    # Check dependencies
                    if self._check_stage_dependencies(file_info, stage):
                        required_stages.append(stage)
            
            plan.append({
                "file_info": file_info,
                "required_stages": required_stages,
                "total_stages": len(ProcessingStage),
                "completed_stages": len(file_info.existing_stages)
            })
        
        return plan
    
    def save_text_file(self, content: str, file_path: Path) -> None:
        """Save text content to a file.
        
        Args:
            content: Text content to save
            file_path: Path to save file
            
        Raises:
            FileWriteError: If file cannot be written
        """
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        except PermissionError as e:
            raise FileWriteError(
                f"Permission denied writing to: {file_path}",
                file_path=str(file_path)
            ) from e
        except Exception as e:
            raise FileWriteError(
                f"Failed to write file: {e}",
                file_path=str(file_path)
            ) from e
    
    def read_text_file(self, file_path: Path) -> str:
        """Read text content from a file.
        
        Args:
            file_path: Path to read from
            
        Returns:
            str: File content
            
        Raises:
            FileNotFoundError: If file doesn't exist
            FileReadError: If file cannot be read
        """
        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}",
                file_path=str(file_path)
            )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except PermissionError as e:
            raise FileReadError(
                f"Permission denied reading: {file_path}",
                file_path=str(file_path)
            ) from e
        except UnicodeDecodeError as e:
            raise FileReadError(
                f"Cannot decode file as UTF-8: {file_path}",
                file_path=str(file_path)
            ) from e
        except Exception as e:
            raise FileReadError(
                f"Failed to read file: {e}",
                file_path=str(file_path)
            ) from e
    
    def ensure_output_directory(self) -> None:
        """Ensure output directory exists and is writable.
        
        Raises:
            FileWriteError: If directory cannot be created or is not writable
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions
            test_file = self.output_dir / ".test_write"
            try:
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                raise FileWriteError(
                    f"Output directory is not writable: {self.output_dir}",
                    file_path=str(self.output_dir)
                ) from e
                
        except PermissionError as e:
            raise FileWriteError(
                f"Cannot create output directory: {self.output_dir}",
                file_path=str(self.output_dir)
            ) from e
    
    def is_supported_audio_format(self, file_path: Path) -> bool:
        """Check if file has supported audio format.
        
        Args:
            file_path: Path to check
            
        Returns:
            bool: True if format is supported
        """
        return file_path.suffix.lower() in self.SUPPORTED_AUDIO_EXTENSIONS
    
    def get_relative_path(self, file_path: Path, base_path: Optional[Path] = None) -> str:
        """Get relative path from base path.
        
        Args:
            file_path: File path to make relative
            base_path: Base path (defaults to output directory)
            
        Returns:
            str: Relative path string
        """
        if base_path is None:
            base_path = self.output_dir
        
        try:
            return str(file_path.relative_to(base_path))
        except ValueError:
            # If relative path cannot be computed, return absolute path
            return str(file_path)
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files in output directory."""
        temp_patterns = ["*.tmp", ".*", ".test_*"]
        
        for pattern in temp_patterns:
            for temp_file in self.output_dir.glob(pattern):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                except Exception:
                    # Ignore cleanup errors
                    pass
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information for output directory.
        
        Returns:
            Dict with storage statistics
        """
        try:
            stat = os.statvfs(self.output_dir)
            total_space = stat.f_frsize * stat.f_blocks
            free_space = stat.f_frsize * stat.f_bavail
            used_space = total_space - free_space
            
            return {
                "total_space": total_space,
                "free_space": free_space,
                "used_space": used_space,
                "usage_percent": (used_space / total_space) * 100 if total_space > 0 else 0
            }
        except (AttributeError, OSError):
            # statvfs not available on Windows, return placeholder
            return {
                "total_space": None,
                "free_space": None,
                "used_space": None,
                "usage_percent": None
            }
    
    def _check_stage_dependencies(self, file_info: FileInfo, stage: ProcessingStage) -> bool:
        """Check if dependencies for a stage are satisfied.
        
        Args:
            file_info: File information
            stage: Stage to check dependencies for
            
        Returns:
            bool: True if dependencies are satisfied
        """
        if stage == ProcessingStage.TRANSCRIPTION:
            # Transcription only needs the audio file
            return file_info.audio_path.exists()
        
        return False