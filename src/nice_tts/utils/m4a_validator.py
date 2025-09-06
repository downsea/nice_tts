"""M4A file validation utilities for nice-tts.

This module provides utilities for validating M4A audio files to prevent
ZeroDivisionError and other issues during transcription processing.
"""

import struct
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import subprocess
import json

from ..core.exceptions import AudioCorruptedError, AudioProcessingError

logger = logging.getLogger('nice_tts')


@dataclass
class ValidationResult:
    """Result of M4A file validation."""
    
    is_valid: bool
    issues: List[str]
    recommended_action: str
    can_be_processed: bool


class M4AValidator:
    """Validator for M4A audio files."""
    
    @staticmethod
    def validate_m4a(file_path: Path) -> ValidationResult:
        """Validate an M4A file for processing.
        
        Args:
            file_path: Path to the M4A file
            
        Returns:
            ValidationResult: Validation result with details
        """
        issues = []
        recommended_action = "process"
        can_be_processed = True
        
        # Check if file exists and is readable
        if not file_path.exists():
            return ValidationResult(
                is_valid=False,
                issues=["File does not exist"],
                recommended_action="skip",
                can_be_processed=False
            )
        
        if not file_path.is_file():
            return ValidationResult(
                is_valid=False,
                issues=["Path is not a file"],
                recommended_action="skip",
                can_be_processed=False
            )
        
        # Check file header
        try:
            has_valid_header = M4AValidator._check_m4a_header(file_path)
            if not has_valid_header:
                issues.append("Invalid M4A header")
                recommended_action = "preprocess"
        except Exception as e:
            logger.warning(f"Error checking M4A header for {file_path}: {e}")
            issues.append(f"Header check failed: {e}")
            recommended_action = "preprocess"
        
        # Check moov atom
        try:
            has_moov = M4AValidator.check_moov_atom(file_path)
            if not has_moov:
                issues.append("Missing or misplaced moov atom")
                recommended_action = "preprocess"
        except Exception as e:
            logger.warning(f"Error checking moov atom for {file_path}: {e}")
            issues.append(f"Moov atom check failed: {e}")
            recommended_action = "preprocess"
        
        # Check duration
        try:
            duration = M4AValidator.get_audio_duration(file_path)
            if duration <= 0:
                issues.append("Zero or negative duration")
                recommended_action = "preprocess"
                can_be_processed = False
        except Exception as e:
            logger.warning(f"Error getting duration for {file_path}: {e}")
            issues.append(f"Duration check failed: {e}")
            recommended_action = "preprocess"
            can_be_processed = False
        
        # Overall validity
        is_valid = len(issues) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            recommended_action=recommended_action,
            can_be_processed=can_be_processed
        )
    
    @staticmethod
    def _check_m4a_header(file_path: Path) -> bool:
        """Check if file has proper M4A header.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if header is valid
        """
        try:
            with open(file_path, 'rb') as f:
                # Read first 12 bytes to check for M4A signature
                header = f.read(12)
                if len(header) < 12:
                    return False
                
                # Check for 'ftyp' box
                box_size = struct.unpack('>I', header[0:4])[0]
                box_type = header[4:8].decode('ascii', errors='ignore')
                
                if box_type != 'ftyp':
                    return False
                
                # Check major brand (should be 'M4A ' or 'mp42' for M4A files)
                major_brand = header[8:12].decode('ascii', errors='ignore')
                if major_brand not in ['M4A ', 'mp42', 'isom']:
                    logger.debug(f"Unrecognized major brand: {major_brand}")
                
                return True
        except Exception as e:
            logger.warning(f"Error reading M4A header: {e}")
            return False
    
    @staticmethod
    def check_moov_atom(file_path: Path) -> bool:
        """Check if moov atom is present and properly positioned.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if moov atom is present
        """
        try:
            # Use ffprobe to check for moov atom
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(file_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.warning(f"ffprobe failed for {file_path}: {result.stderr}")
                return False
            
            data = json.loads(result.stdout)
            
            # Check if we have valid format and streams
            if 'format' not in data or 'streams' not in data:
                return False
            
            # Check if we have at least one audio stream
            audio_streams = [s for s in data['streams'] if s.get('codec_type') == 'audio']
            if not audio_streams:
                return False
            
            return True
        except subprocess.TimeoutExpired:
            logger.warning(f"ffprobe timed out for {file_path}")
            return False
        except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Error checking moov atom with ffprobe: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error checking moov atom: {e}")
            return False
    
    @staticmethod
    def get_audio_duration(file_path: Path) -> float:
        """Get audio duration, handling edge cases.
        
        Args:
            file_path: Path to the file
            
        Returns:
            float: Audio duration in seconds
            
        Raises:
            AudioProcessingError: If duration cannot be determined
        """
        try:
            # Use ffprobe to get duration
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                str(file_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise AudioProcessingError(
                    f"Failed to get audio duration: ffprobe error",
                    audio_path=str(file_path)
                )
            
            data = json.loads(result.stdout)
            
            # Try to get duration from format
            duration_str = data.get('format', {}).get('duration')
            if duration_str:
                duration = float(duration_str)
                if duration > 0:
                    return duration
            
            # Try to get duration from streams
            streams = data.get('streams', [])
            for stream in streams:
                if stream.get('codec_type') == 'audio':
                    duration_str = stream.get('duration')
                    if duration_str:
                        duration = float(duration_str)
                        if duration > 0:
                            return duration
            
            # If we get here, we couldn't determine a valid duration
            raise AudioProcessingError(
                "Could not determine audio duration",
                audio_path=str(file_path)
            )
            
        except subprocess.TimeoutExpired:
            raise AudioProcessingError(
                "Timeout getting audio duration",
                audio_path=str(file_path)
            )
        except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError) as e:
            raise AudioProcessingError(
                f"Failed to get audio duration: {e}",
                audio_path=str(file_path)
            ) from e
        except Exception as e:
            raise AudioProcessingError(
                f"Unexpected error getting audio duration: {e}",
                audio_path=str(file_path)
            ) from e