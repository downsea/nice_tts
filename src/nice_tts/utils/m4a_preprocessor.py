"""M4A file preprocessing utilities for nice-tts.

This module provides utilities for repairing and converting problematic M4A files
to prevent transcription errors.
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ..core.exceptions import AudioProcessingError

logger = logging.getLogger('nice_tts')


class M4APreprocessor:
    """Preprocessor for M4A audio files."""
    
    @staticmethod
    def repair_m4a(input_path: Path, output_path: Path) -> bool:
        """Repair an M4A file by re-encoding it.
        
        Uses ffmpeg to fix moov atom positioning and other issues.
        
        Args:
            input_path: Path to the input M4A file
            output_path: Path to save the repaired file
            
        Returns:
            bool: True if repair was successful
        """
        try:
            # Use ffmpeg to re-encode the file, which will fix moov atom positioning
            cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-c:a', 'copy',  # Copy audio stream without re-encoding
                '-movflags', '+faststart',  # Optimize for streaming (fixes moov atom)
                '-y',  # Overwrite output file
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.warning(f"ffmpeg repair failed for {input_path}: {result.stderr}")
                return False
            
            logger.info(f"Successfully repaired M4A file: {input_path} -> {output_path}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"ffmpeg repair timed out for {input_path}")
            return False
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.error(f"Error repairing M4A file {input_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error repairing M4A file {input_path}: {e}")
            return False
    
    @staticmethod
    def convert_to_wav(input_path: Path, output_path: Path) -> bool:
        """Convert M4A to WAV as fallback option.
        
        Args:
            input_path: Path to the input M4A file
            output_path: Path to save the WAV file
            
        Returns:
            bool: True if conversion was successful
        """
        try:
            # Use ffmpeg to convert M4A to WAV
            cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-ar', '16000',  # Set sample rate to 16kHz (standard for Whisper)
                '-ac', '1',      # Set to mono
                '-c:a', 'pcm_s16le',  # Use 16-bit PCM encoding
                '-y',  # Overwrite output file
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.warning(f"ffmpeg conversion failed for {input_path}: {result.stderr}")
                return False
            
            logger.info(f"Successfully converted M4A to WAV: {input_path} -> {output_path}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"ffmpeg conversion timed out for {input_path}")
            return False
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.error(f"Error converting M4A file {input_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error converting M4A file {input_path}: {e}")
            return False
    
    @staticmethod
    def create_temp_repaired_file(input_path: Path) -> Optional[Path]:
        """Create a temporary repaired version of an M4A file.
        
        Args:
            input_path: Path to the input M4A file
            
        Returns:
            Path: Path to temporary repaired file, or None if failed
        """
        try:
            temp_dir = Path(tempfile.gettempdir())
            temp_file = temp_dir / f"nice_tts_repaired_{input_path.stem}.m4a"
            
            success = M4APreprocessor.repair_m4a(input_path, temp_file)
            if success:
                return temp_file
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error creating temporary repaired file for {input_path}: {e}")
            return None
    
    @staticmethod
    def create_temp_wav_file(input_path: Path) -> Optional[Path]:
        """Create a temporary WAV version of an M4A file.
        
        Args:
            input_path: Path to the input M4A file
            
        Returns:
            Path: Path to temporary WAV file, or None if failed
        """
        try:
            temp_dir = Path(tempfile.gettempdir())
            temp_file = temp_dir / f"nice_tts_converted_{input_path.stem}.wav"
            
            success = M4APreprocessor.convert_to_wav(input_path, temp_file)
            if success:
                return temp_file
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error creating temporary WAV file for {input_path}: {e}")
            return None