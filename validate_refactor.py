"""Simple validation script to test the refactored nice-tts system.

This script tests basic functionality without requiring external dependencies
like typer, transformers, etc.
"""

import sys
from pathlib import Path
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all major modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test core modules
        from nice_tts.core.config import TranscriptionConfig, LLMConfig, OutputConfig, LoggingConfig, AppConfig
        from nice_tts.core.exceptions import NiceTTSError, ConfigurationError, ValidationError
        print("✅ Core modules import successfully")
        
        # Test engine base classes
        from nice_tts.engines.transcription.base import TranscriptionEngine, TranscriptionResult
        from nice_tts.engines.llm.base import LLMEngine, RefinementResult, SummaryResult
        print("✅ Engine base classes import successfully")
        
        # Test utilities
        from nice_tts.utils.file_manager import FileManager, ProcessingStage
        from nice_tts.utils.logger import Logger, LogLevel
        print("✅ Utility modules import successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration system...")
    
    try:
        from nice_tts.core.config import TranscriptionConfig, LLMConfig, OutputConfig, AppConfig
        
        # Test default configurations
        transcription_config = TranscriptionConfig()
        assert transcription_config.model_name == "large-v3-turbo"
        assert transcription_config.language == "zh"
        
        llm_config = LLMConfig()
        assert llm_config.provider == "openai"
        assert llm_config.max_tokens == 128000
        
        output_config = OutputConfig()
        assert output_config.directory == Path("output")
        
        app_config = AppConfig()
        assert app_config.parallel_jobs == 1
        
        print("✅ Configuration classes work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_file_manager():
    """Test file management system."""
    print("\nTesting file management system...")
    
    try:
        from nice_tts.utils.file_manager import FileManager, ProcessingStage
        from nice_tts.core.config import OutputConfig
        import tempfile
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Test FileManager initialization
            config = OutputConfig(directory=tmp_path / "output")
            
            # Mock the ensure_output_directory method to avoid actual file operations
            class MockFileManager(FileManager):
                def ensure_output_directory(self):
                    pass
            
            manager = MockFileManager(config)
            
            # Test supported formats
            assert manager.is_supported_audio_format(Path("test.wav"))
            assert manager.is_supported_audio_format(Path("test.mp3"))
            assert not manager.is_supported_audio_format(Path("test.txt"))
            
            print("✅ File management system works correctly")
            return True
        
    except Exception as e:
        print(f"❌ File management test failed: {e}")
        return False

def test_exceptions():
    """Test exception system."""
    print("\nTesting exception system...")
    
    try:
        from nice_tts.core.exceptions import (
            NiceTTSError, ConfigurationError, ValidationError,
            FileOperationError, TranscriptionError, LLMError
        )
        
        # Test basic exception creation
        error = NiceTTSError("Test error", {"key": "value"})
        assert str(error) == "Test error (Details: {'key': 'value'})"
        
        # Test specific exceptions
        config_error = ConfigurationError("Config error")
        assert isinstance(config_error, NiceTTSError)
        
        file_error = FileOperationError("File error", "/path/to/file")
        assert "/path/to/file" in str(file_error)
        
        print("✅ Exception system works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Exception test failed: {e}")
        return False

def test_processing_stages():
    """Test processing stage enumeration."""
    print("\nTesting processing stages...")
    
    try:
        from nice_tts.utils.file_manager import ProcessingStage
        
        stages = list(ProcessingStage)
        expected_stages = ["transcription", "refinement", "summarization"]
        
        assert len(stages) == 3
        for stage in stages:
            assert stage.value in expected_stages
        
        print("✅ Processing stages work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Processing stage test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🔍 Validating Nice-TTS Refactored System")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_configuration,
        test_file_manager,
        test_exceptions,
        test_processing_stages
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Validation Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All validation tests passed! The refactored system is working correctly.")
        return True
    else:
        print(f"⚠️  {failed} validation tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)