"""Test file management functionality."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import tempfile
import shutil

from nice_tts.utils.file_manager import (
    FileManager, FileInfo, ProcessingStage
)
from nice_tts.core.config import OutputConfig
from nice_tts.core.exceptions import (
    FileNotFoundError, ValidationError, FileWriteError, FileReadError
)


class TestProcessingStage:
    """Test ProcessingStage enum."""
    
    def test_stage_values(self):
        """Test enum values."""
        assert ProcessingStage.TRANSCRIPTION.value == "transcription"
        assert ProcessingStage.REFINEMENT.value == "refinement"
        assert ProcessingStage.SUMMARIZATION.value == "summarization"


class TestFileInfo:
    """Test FileInfo class."""
    
    def test_from_audio_path_basic(self, tmp_path):
        """Test FileInfo creation from audio path."""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        file_info = FileInfo.from_audio_path(audio_path, output_dir)
        
        assert file_info.audio_path == audio_path
        assert file_info.output_dir == output_dir
        assert file_info.base_name == "test"
        assert file_info.transcript_path == output_dir / "test.txt"
        assert file_info.refined_path == output_dir / "test.fine.txt"
        assert file_info.summary_path == output_dir / "test.md"
        assert len(file_info.existing_stages) == 0
        
    def test_from_audio_path_with_existing_files(self, tmp_path):
        """Test FileInfo with existing output files."""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Create existing output files
        (output_dir / "test.txt").write_text("transcript")
        (output_dir / "test.fine.txt").write_text("refined")
        
        file_info = FileInfo.from_audio_path(audio_path, output_dir)
        
        assert ProcessingStage.TRANSCRIPTION in file_info.existing_stages
        assert ProcessingStage.REFINEMENT in file_info.existing_stages
        assert ProcessingStage.SUMMARIZATION not in file_info.existing_stages
        
    def test_needs_processing(self, tmp_path):
        """Test needs_processing method."""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        file_info = FileInfo.from_audio_path(audio_path, output_dir)
        
        # No files exist, should need processing
        assert file_info.needs_processing(ProcessingStage.TRANSCRIPTION) is True
        assert file_info.needs_processing(ProcessingStage.REFINEMENT) is True
        
        # With force=True, should always need processing
        assert file_info.needs_processing(ProcessingStage.TRANSCRIPTION, force=True) is True
        
    def test_get_stage_path(self, tmp_path):
        """Test get_stage_path method."""
        audio_path = tmp_path / "test.wav"
        output_dir = tmp_path / "output"
        
        file_info = FileInfo.from_audio_path(audio_path, output_dir)
        
        assert file_info.get_stage_path(ProcessingStage.TRANSCRIPTION) == file_info.transcript_path
        assert file_info.get_stage_path(ProcessingStage.REFINEMENT) == file_info.refined_path
        assert file_info.get_stage_path(ProcessingStage.SUMMARIZATION) == file_info.summary_path
        
    def test_get_input_for_stage(self, tmp_path):
        """Test get_input_for_stage method.""" 
        audio_path = tmp_path / "test.wav"
        audio_path.touch()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Create transcript file
        (output_dir / "test.txt").write_text("transcript")
        
        file_info = FileInfo.from_audio_path(audio_path, output_dir)
        
        assert file_info.get_input_for_stage(ProcessingStage.TRANSCRIPTION) == audio_path
        assert file_info.get_input_for_stage(ProcessingStage.REFINEMENT) == file_info.transcript_path
        # Summarization should return None since refined file doesn't exist
        assert file_info.get_input_for_stage(ProcessingStage.SUMMARIZATION) is None


class TestFileManager:
    """Test FileManager class."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.config = OutputConfig(directory=Path("test_output"))
        
    def test_init(self):
        """Test FileManager initialization.""" 
        with patch.object(FileManager, 'ensure_output_directory'):
            manager = FileManager(self.config)
            assert manager.config == self.config
            assert manager.output_dir == Path("test_output")
            
    def test_discover_audio_files_single_file(self, tmp_path):
        """Test discovering single audio file."""
        audio_file = tmp_path / "test.wav"
        audio_file.touch()
        
        config = OutputConfig(directory=tmp_path / "output")
        
        with patch.object(FileManager, 'ensure_output_directory'):
            manager = FileManager(config)
            
        files = manager.discover_audio_files(audio_file)
        assert len(files) == 1
        assert files[0] == audio_file
        
    def test_discover_audio_files_directory(self, tmp_path):
        """Test discovering audio files in directory."""
        # Create audio files
        audio_files = [
            tmp_path / "test1.wav",
            tmp_path / "test2.mp3",
            tmp_path / "test3.m4a",
            tmp_path / "not_audio.txt"  # Should be ignored
        ]
        
        for f in audio_files:
            f.touch()
            
        config = OutputConfig(directory=tmp_path / "output")
        
        with patch.object(FileManager, 'ensure_output_directory'):
            manager = FileManager(config)
            
        files = manager.discover_audio_files(tmp_path)
        
        # Should find 3 audio files, sorted
        assert len(files) == 3
        assert all(f.suffix.lower() in manager.SUPPORTED_AUDIO_EXTENSIONS for f in files)
        
    def test_discover_audio_files_not_found(self):
        """Test file not found error."""
        config = OutputConfig(directory=Path("test_output"))
        
        with patch.object(FileManager, 'ensure_output_directory'):
            manager = FileManager(config)
            
        with pytest.raises(FileNotFoundError):
            manager.discover_audio_files(Path("nonexistent.wav"))
            
    def test_discover_audio_files_unsupported_format(self, tmp_path):
        """Test unsupported audio format."""
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.touch()
        
        config = OutputConfig(directory=tmp_path / "output")
        
        with patch.object(FileManager, 'ensure_output_directory'):
            manager = FileManager(config)
            
        with pytest.raises(ValidationError, match="Unsupported audio format"):
            manager.discover_audio_files(unsupported_file)
            
    def test_discover_audio_files_no_audio_in_dir(self, tmp_path):
        """Test directory with no audio files."""
        text_file = tmp_path / "readme.txt"
        text_file.touch()
        
        config = OutputConfig(directory=tmp_path / "output")
        
        with patch.object(FileManager, 'ensure_output_directory'):
            manager = FileManager(config)
            
        with pytest.raises(ValidationError, match="No supported audio files found"):
            manager.discover_audio_files(tmp_path)
            
    def test_save_text_file(self, tmp_path):
        """Test saving text file."""
        config = OutputConfig(directory=tmp_path / "output")
        
        with patch.object(FileManager, 'ensure_output_directory'):
            manager = FileManager(config)
            
        test_content = "Hello, world!"
        test_path = tmp_path / "test.txt"
        
        manager.save_text_file(test_content, test_path)
        
        assert test_path.exists()
        assert test_path.read_text(encoding='utf-8') == test_content
        
    def test_read_text_file(self, tmp_path):
        """Test reading text file."""
        config = OutputConfig(directory=tmp_path / "output")
        
        with patch.object(FileManager, 'ensure_output_directory'):
            manager = FileManager(config)
            
        test_content = "Hello, world!"
        test_path = tmp_path / "test.txt"
        test_path.write_text(test_content, encoding='utf-8')
        
        content = manager.read_text_file(test_path)
        assert content == test_content
        
    def test_read_text_file_not_found(self, tmp_path):
        """Test reading non-existent file."""
        config = OutputConfig(directory=tmp_path / "output")
        
        with patch.object(FileManager, 'ensure_output_directory'):
            manager = FileManager(config)
            
        with pytest.raises(FileNotFoundError):
            manager.read_text_file(tmp_path / "nonexistent.txt")
            
    def test_is_supported_audio_format(self):
        """Test audio format checking."""
        config = OutputConfig(directory=Path("test_output"))
        
        with patch.object(FileManager, 'ensure_output_directory'):
            manager = FileManager(config)
            
        assert manager.is_supported_audio_format(Path("test.wav")) is True
        assert manager.is_supported_audio_format(Path("test.MP3")) is True
        assert manager.is_supported_audio_format(Path("test.txt")) is False
        
    def test_get_processing_plan(self, tmp_path):
        """Test processing plan generation."""
        # Setup audio files
        audio1 = tmp_path / "audio1.wav"
        audio2 = tmp_path / "audio2.wav"
        audio1.touch()
        audio2.touch()
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        config = OutputConfig(directory=output_dir)
        
        with patch.object(FileManager, 'ensure_output_directory'):
            manager = FileManager(config)
            
        plan = manager.get_processing_plan([audio1, audio2])
        
        assert len(plan) == 2
        assert all("file_info" in item for item in plan)
        assert all("required_stages" in item for item in plan)
        
        # All stages should be required since no output files exist
        for item in plan:
            assert len(item["required_stages"]) == 3  # All 3 stages
            
    def test_get_processing_plan_with_existing_files(self, tmp_path):
        """Test processing plan with existing output files."""
        audio1 = tmp_path / "audio1.wav"
        audio1.touch()
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Create existing transcript
        (output_dir / "audio1.txt").write_text("existing transcript")
        
        config = OutputConfig(directory=output_dir)
        
        with patch.object(FileManager, 'ensure_output_directory'):
            manager = FileManager(config)
            
        plan = manager.get_processing_plan([audio1])
        
        assert len(plan) == 1
        
        # Only refinement and summarization should be required
        required_stages = plan[0]["required_stages"]
        assert ProcessingStage.TRANSCRIPTION not in required_stages
        assert ProcessingStage.REFINEMENT in required_stages
        assert ProcessingStage.SUMMARIZATION in required_stages


# Integration tests
class TestFileManagerIntegration:
    """Integration tests for FileManager."""
    
    def test_full_workflow(self, tmp_path):
        """Test complete file management workflow."""
        # Setup
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        output_dir = tmp_path / "output"
        
        audio_files = [
            audio_dir / "meeting1.wav",
            audio_dir / "meeting2.mp3"
        ]
        
        for f in audio_files:
            f.touch()
            
        config = OutputConfig(directory=output_dir)
        manager = FileManager(config)
        
        # Discover files
        discovered = manager.discover_audio_files(audio_dir)
        assert len(discovered) == 2
        
        # Create processing plan
        plan = manager.get_processing_plan(discovered)
        assert len(plan) == 2
        
        # Simulate processing one file
        file_info = plan[0]["file_info"]
        
        # Save transcript
        manager.save_text_file("Test transcript", file_info.transcript_path)
        
        # Verify file exists and can be read
        content = manager.read_text_file(file_info.transcript_path)
        assert content == "Test transcript"
        
        # Update plan should show transcript stage complete
        updated_plan = manager.get_processing_plan(discovered)
        file1_plan = next(item for item in updated_plan 
                         if item["file_info"].audio_path == file_info.audio_path)
        
        assert ProcessingStage.TRANSCRIPTION not in file1_plan["required_stages"]


if __name__ == "__main__":
    pytest.main([__file__])