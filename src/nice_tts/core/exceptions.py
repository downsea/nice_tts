"""Exception definitions for nice-tts.

This module provides a hierarchy of custom exceptions for different error scenarios
that can occur during audio processing, transcription, and LLM operations.
"""

from typing import Optional, Any, Dict


class NiceTTSError(Exception):
    """Base exception class for all nice-tts errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class ConfigurationError(NiceTTSError):
    """Raised when there are configuration-related errors."""
    pass


class ValidationError(NiceTTSError):
    """Raised when input validation fails."""
    pass


class FileOperationError(NiceTTSError):
    """Base class for file-related errors."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.file_path = file_path
        
    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.file_path:
            return f"{base_msg} (File: {self.file_path})"
        return base_msg


class FileNotFoundError(FileOperationError):
    """Raised when a required file is not found."""
    pass


class FileReadError(FileOperationError):
    """Raised when a file cannot be read."""
    pass


class FileWriteError(FileOperationError):
    """Raised when a file cannot be written."""
    pass


class AudioProcessingError(NiceTTSError):
    """Base class for audio processing errors."""
    
    def __init__(self, message: str, audio_path: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.audio_path = audio_path
        
    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.audio_path:
            return f"{base_msg} (Audio: {self.audio_path})"
        return base_msg


class UnsupportedAudioFormatError(AudioProcessingError):
    """Raised when an unsupported audio format is encountered."""
    pass


class AudioCorruptedError(AudioProcessingError):
    """Raised when audio file is corrupted or cannot be decoded."""
    pass


class TranscriptionError(NiceTTSError):
    """Base class for transcription-related errors."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.model_name = model_name
        
    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.model_name:
            return f"{base_msg} (Model: {self.model_name})"
        return base_msg


class ModelLoadError(TranscriptionError):
    """Raised when a transcription model fails to load."""
    pass


class ModelNotFoundError(TranscriptionError):
    """Raised when a transcription model is not found."""
    pass


class TranscriptionFailureError(TranscriptionError):
    """Raised when transcription process fails."""
    pass


class DeviceError(TranscriptionError):
    """Raised when there are GPU/device-related issues."""
    pass


class LLMError(NiceTTSError):
    """Base class for LLM-related errors."""
    
    def __init__(self, message: str, provider: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.provider = provider
        
    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.provider:
            return f"{base_msg} (Provider: {self.provider})"
        return base_msg


class APIError(LLMError):
    """Raised when LLM API calls fail."""
    
    def __init__(self, message: str, provider: Optional[str] = None, status_code: Optional[int] = None, 
                 response_body: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, provider, details)
        self.status_code = status_code
        self.response_body = response_body
        
    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.status_code:
            base_msg += f" (Status: {self.status_code})"
        return base_msg


class APIRateLimitError(APIError):
    """Raised when API rate limits are exceeded."""
    pass


class APIAuthenticationError(APIError):
    """Raised when API authentication fails."""
    pass


class APIQuotaExceededError(APIError):
    """Raised when API quota is exceeded."""
    pass


class TokenLimitError(LLMError):
    """Raised when text exceeds token limits."""
    
    def __init__(self, message: str, token_count: int, limit: int, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details=details)
        self.token_count = token_count
        self.limit = limit
        
    def __str__(self) -> str:
        return f"{self.message} (Tokens: {self.token_count}/{self.limit})"
    
    @classmethod
    def from_tokenizer_error(cls, original_error: Exception, text_length: int, limit: int) -> "TokenLimitError":
        """Create TokenLimitError from tokenizer exception.
        
        Args:
            original_error: The original tokenizer exception
            text_length: Length of the text that caused the error
            limit: The token limit that was exceeded
            
        Returns:
            TokenLimitError: New exception with context
        """
        return cls(
            f"Token limit exceeded during tokenization: {original_error}",
            token_count=text_length // 2,  # Rough estimate
            limit=limit,
            details={
                "original_error": str(original_error),
                "error_type": type(original_error).__name__,
                "text_length_chars": text_length,
                "suggested_action": "enable_chunking"
            }
        )


class NetworkError(NiceTTSError):
    """Raised when network operations fail."""
    
    def __init__(self, message: str, url: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.url = url
        
    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.url:
            return f"{base_msg} (URL: {self.url})"
        return base_msg


class TimeoutError(NetworkError):
    """Raised when operations timeout."""
    pass


class ProcessingError(NiceTTSError):
    """Base class for processing pipeline errors."""
    
    def __init__(self, message: str, stage: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.stage = stage
        
    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.stage:
            return f"{base_msg} (Stage: {self.stage})"
        return base_msg


class PipelineError(ProcessingError):
    """Raised when the processing pipeline fails."""
    pass


class RetryableError(NiceTTSError):
    """Base class for errors that can be retried."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.retry_after = retry_after  # seconds to wait before retry
        
    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.retry_after:
            return f"{base_msg} (Retry after: {self.retry_after}s)"
        return base_msg


class ChunkingError(ProcessingError):
    """Raised when text chunking fails."""
    
    def __init__(self, message: str, text_length: int, max_tokens: int, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, stage="chunking", details=details)
        self.text_length = text_length
        self.max_tokens = max_tokens
        
    def __str__(self) -> str:
        return f"{self.message} (Text length: {self.text_length}, Max tokens: {self.max_tokens})"


class TokenizerError(LLMError):
    """Raised when tokenizer operations fail."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details=details)
        self.model_name = model_name
        
    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.model_name:
            return f"{base_msg} (Model: {self.model_name})"
        return base_msg


class FatalError(NiceTTSError):
    """Raised for non-recoverable errors that should stop processing."""
    pass


# Utility functions for error handling

def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable."""
    retryable_types = (
        RetryableError,
        NetworkError,
        TimeoutError,
        APIRateLimitError,
    )
    return isinstance(error, retryable_types)


def get_retry_delay(error: Exception) -> int:
    """Get suggested retry delay for an error."""
    if isinstance(error, RetryableError) and error.retry_after:
        return error.retry_after
    elif isinstance(error, APIRateLimitError):
        return 60  # Standard rate limit backoff
    elif isinstance(error, NetworkError):
        return 5   # Network error backoff
    else:
        return 1   # Default backoff


def create_error_from_exception(
    exc: Exception, 
    context: Optional[Dict[str, Any]] = None
) -> NiceTTSError:
    """Convert a standard exception to a nice-tts error with context."""
    context = context or {}
    
    if isinstance(exc, FileNotFoundError):
        return FileNotFoundError(
            f"File not found: {exc}",
            file_path=context.get("file_path"),
            details=context
        )
    elif isinstance(exc, PermissionError):
        return FileOperationError(
            f"Permission denied: {exc}",
            details=context
        )
    elif isinstance(exc, ConnectionError):
        return NetworkError(
            f"Network connection failed: {exc}",
            url=context.get("url"),
            details=context
        )
    elif isinstance(exc, TimeoutError):
        return TimeoutError(
            f"Operation timed out: {exc}",
            url=context.get("url"),
            details=context
        )
    else:
        return NiceTTSError(
            f"Unexpected error: {exc}",
            details={**context, "original_type": type(exc).__name__}
        )