"""Base classes for LLM engines.

This module defines the abstract interface that all LLM engines must implement,
providing a consistent API for different LLM providers like OpenAI and Claude.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import time

from ...core.config import LLMConfig


@dataclass
class RefinementResult:
    """Result of a text refinement operation."""
    
    refined_text: str
    chunks_processed: int
    tokens_used: int
    processing_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate result data."""
        if not isinstance(self.refined_text, str):
            raise ValueError("refined_text must be a string")
        if self.chunks_processed <= 0:
            raise ValueError("chunks_processed must be positive")
        if self.tokens_used < 0:
            raise ValueError("tokens_used must be non-negative")
        if self.processing_time < 0:
            raise ValueError("processing_time must be non-negative")


@dataclass
class SummaryResult:
    """Result of a text summarization operation."""
    
    summary_markdown: str
    tokens_used: int
    processing_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate result data."""
        if not isinstance(self.summary_markdown, str):
            raise ValueError("summary_markdown must be a string")
        if self.tokens_used < 0:
            raise ValueError("tokens_used must be non-negative")
        if self.processing_time < 0:
            raise ValueError("processing_time must be non-negative")


class LLMEngine(ABC):
    """Abstract base class for LLM engines."""
    
    def __init__(self, config: LLMConfig):
        """Initialize the LLM engine.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        self._client = None
        self._tokenizer = None
    
    @abstractmethod
    def refine_text(self, text: str) -> RefinementResult:
        """Refine transcribed text using LLM.
        
        Args:
            text: Raw transcribed text to refine
            
        Returns:
            RefinementResult: The refinement result
            
        Raises:
            LLMError: If refinement fails
            TokenLimitError: If text exceeds token limits
            APIError: If API call fails
        """
        pass
    
    @abstractmethod
    def summarize_text(
        self, 
        text: str, 
        audio_path: Optional[str] = None,
        original_txt_path: Optional[str] = None,
        refined_txt_path: Optional[str] = None
    ) -> SummaryResult:
        """Generate summary from refined text.
        
        Args:
            text: Refined text to summarize
            audio_path: Path to original audio file (for metadata)
            original_txt_path: Path to original transcript (for linking)
            refined_txt_path: Path to refined transcript (for linking)
            
        Returns:
            SummaryResult: The summary result
            
        Raises:
            LLMError: If summarization fails
            TokenLimitError: If text exceeds token limits
            APIError: If API call fails
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            int: Number of tokens
        """
        pass
    
    @abstractmethod
    def get_max_tokens(self) -> int:
        """Get maximum token limit for this engine.
        
        Returns:
            int: Maximum tokens supported
        """
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate that the LLM service is accessible.
        
        Returns:
            bool: True if connection is valid
            
        Raises:
            APIAuthenticationError: If authentication fails
            NetworkError: If connection fails
        """
        pass
    
    def chunk_text(self, text: str, max_tokens: Optional[int] = None) -> List[str]:
        """Split text into chunks that fit within token limits.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk (defaults to config max_tokens)
            
        Returns:
            List[str]: List of text chunks
        """
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        
        # If text fits in one chunk, return as-is
        if self.count_tokens(text) <= max_tokens:
            return [text]
        
        return self._split_text_intelligently(text, max_tokens)
    
    def _split_text_intelligently(self, text: str, max_tokens: int) -> List[str]:
        """Split text at natural boundaries while respecting token limits.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List[str]: List of text chunks
        """
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first to maintain context
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            # Check if adding this paragraph would exceed token limit
            test_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
            
            if self.count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Handle oversized paragraphs
                if self.count_tokens(paragraph) > max_tokens:
                    # Split paragraph by sentences
                    sentence_chunks = self._split_paragraph_by_sentences(paragraph, max_tokens)
                    chunks.extend(sentence_chunks[:-1])  # Add all but last
                    current_chunk = sentence_chunks[-1] if sentence_chunks else ""
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_paragraph_by_sentences(self, paragraph: str, max_tokens: int) -> List[str]:
        """Split a paragraph by sentences while respecting token limits."""
        import re
        
        # Simple sentence splitting (can be improved)
        sentences = re.split(r'[.!?]+\s+', paragraph)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            sentence = sentence.strip() + "."
            test_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if self.count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If single sentence is too long, force split
                if self.count_tokens(sentence) > max_tokens:
                    words = sentence.split()
                    word_chunk = ""
                    
                    for word in words:
                        test_word_chunk = word_chunk + (" " if word_chunk else "") + word
                        
                        if self.count_tokens(test_word_chunk) <= max_tokens:
                            word_chunk = test_word_chunk
                        else:
                            if word_chunk:
                                chunks.append(word_chunk)
                            word_chunk = word
                    
                    current_chunk = word_chunk
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks


class LLMEngineRegistry:
    """Registry for LLM engines."""
    
    def __init__(self):
        self._engines: Dict[str, type] = {}
    
    def register(self, name: str, engine_class: type) -> None:
        """Register an LLM engine.
        
        Args:
            name: Engine name (e.g., 'openai', 'claude')
            engine_class: Engine class that implements LLMEngine
        """
        if not issubclass(engine_class, LLMEngine):
            raise ValueError("Engine class must inherit from LLMEngine")
        
        self._engines[name] = engine_class
    
    def get(self, name: str) -> type:
        """Get a registered engine class.
        
        Args:
            name: Engine name
            
        Returns:
            Engine class
            
        Raises:
            KeyError: If engine is not registered
        """
        if name not in self._engines:
            raise KeyError(f"Unknown LLM engine: {name}")
        
        return self._engines[name]
    
    def list_engines(self) -> List[str]:
        """Get list of registered engine names."""
        return list(self._engines.keys())
    
    def create_engine(self, name: str, config: LLMConfig) -> LLMEngine:
        """Create an instance of an LLM engine.
        
        Args:
            name: Engine name
            config: LLM configuration
            
        Returns:
            LLMEngine instance
        """
        engine_class = self.get(name)
        return engine_class(config)


# Global registry instance
registry = LLMEngineRegistry()