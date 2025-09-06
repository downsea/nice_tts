"""Base classes for LLM engines.

This module defines the abstract interface that all LLM engines must implement,
providing a consistent API for different LLM providers like OpenAI and Ollama.
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
        
        # Apply safety margin from config
        safe_max_tokens = int(max_tokens * self.config.safety_margin)
        
        # Check if chunking is disabled in config
        if not getattr(self.config, 'enable_chunking', True):
            return [text]  # Return as single chunk
        
        # Pre-check to avoid expensive operations on small texts
        estimated_tokens = self._quick_token_estimate(text)
        if estimated_tokens <= safe_max_tokens:
            # Double-check with accurate count for borderline cases
            actual_tokens = self.count_tokens(text)
            if actual_tokens <= safe_max_tokens:
                return [text]
        
        return self._split_text_intelligently(text, safe_max_tokens)
    
    def chunk_text_with_overlap(self, text: str, max_tokens: Optional[int] = None) -> List[str]:
        """Split text into overlapping chunks for better context preservation.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List[str]: List of overlapping text chunks
        """
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        
        # Get overlap size from config
        overlap_tokens = getattr(self.config, 'chunk_overlap', 100)
        safe_max_tokens = int(max_tokens * self.config.safety_margin)
        
        # Get base chunks first
        base_chunks = self._split_text_intelligently(text, safe_max_tokens)
        
        # If only one chunk or overlap disabled, return as-is
        if len(base_chunks) <= 1 or overlap_tokens <= 0:
            return base_chunks
        
        # Add overlap between chunks
        overlapped_chunks = []
        for i, chunk in enumerate(base_chunks):
            if i == 0:
                # First chunk remains unchanged
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = base_chunks[i - 1]
                overlap_text = self._extract_overlap(prev_chunk, overlap_tokens)
                
                if overlap_text:
                    combined_chunk = overlap_text + "\n\n" + chunk
                    # Ensure combined chunk doesn't exceed limits
                    if self.count_tokens(combined_chunk) <= safe_max_tokens:
                        overlapped_chunks.append(combined_chunk)
                    else:
                        overlapped_chunks.append(chunk)  # Skip overlap if too large
                else:
                    overlapped_chunks.append(chunk)
        
        return overlapped_chunks
    
    def _extract_overlap(self, text: str, overlap_tokens: int) -> str:
        """Extract the last N tokens from text for overlap.
        
        Args:
            text: Source text
            overlap_tokens: Number of tokens to extract
            
        Returns:
            str: Overlap text
        """
        # Simple approach: take last few sentences
        import re
        sentences = re.split(r'[.!?]+\s+', text)
        
        overlap_text = ""
        for sentence in reversed(sentences):
            test_overlap = sentence + (" " + overlap_text if overlap_text else "")
            if self.count_tokens(test_overlap) <= overlap_tokens:
                overlap_text = test_overlap
            else:
                break
        
        return overlap_text
    
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
        
        # Post-process chunks to ensure none exceed limits
        final_chunks = []
        for chunk in chunks:
            if self.count_tokens(chunk) > max_tokens:
                # Use emergency splitting for oversized chunks
                sub_chunks = self._split_paragraph_by_sentences(chunk, max_tokens)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _split_paragraph_by_sentences(self, paragraph: str, max_tokens: int) -> List[str]:
        """Split a paragraph by sentences while respecting token limits."""
        import re
        
        # Improved sentence splitting with better punctuation handling
        sentence_endings = r'[.!?。！？;；]'
        sentences = re.split(f'({sentence_endings}+)', paragraph)
        
        # Reconstruct sentences with their punctuation
        reconstructed_sentences = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and re.match(sentence_endings, sentences[i + 1]):
                # This is punctuation, combine with previous text
                sentence = sentences[i] + sentences[i + 1]
                i += 2
            else:
                sentence = sentences[i]
                i += 1
            
            if sentence.strip():
                reconstructed_sentences.append(sentence.strip())
        
        chunks = []
        current_chunk = ""
        
        for sentence in reconstructed_sentences:
            if not sentence.strip():
                continue
            
            sentence = sentence.strip()
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
        
        # Final fallback: if still no chunks, split by character limit
        if not chunks and paragraph:
            char_limit = max_tokens * 3  # Conservative estimate
            chunks = self._fallback_char_split(paragraph, char_limit)
        
        return chunks if chunks else [paragraph[:max_tokens * 3]]  # Ultimate fallback
    
    def _fallback_char_split(self, text: str, char_limit: int) -> List[str]:
        """Fallback character-based splitting when token counting fails.
        
        Args:
            text: Text to split
            char_limit: Character limit per chunk
            
        Returns:
            List[str]: Character-split chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + char_limit
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at sentence boundary
            boundary_chars = ['。', '.', '!', '?', '！', '？', ';', '；']
            best_break = end
            
            for i in range(end - 100, start, -1):  # Look back up to 100 chars
                if text[i] in boundary_chars:
                    best_break = i + 1
                    break
            
            chunks.append(text[start:best_break])
            start = best_break
        
        return chunks
    
    def _quick_token_estimate(self, text: str) -> int:
        """Quick and safe token estimation without using tokenizer.
        
        This method provides a rough but safe overestimate to avoid
        expensive tokenizer calls on very long texts.
        
        Args:
            text: Text to estimate
            
        Returns:
            int: Estimated token count (intentionally overestimates)
        """
        if not text:
            return 0
        
        # Count different types of characters for more accurate estimation
        chinese_chars = 0
        english_chars = 0
        punctuation_count = 0
        
        import re
        
        # Count Chinese characters (more token-dense)
        chinese_pattern = r'[\u4e00-\u9fff]'
        chinese_chars = len(re.findall(chinese_pattern, text))
        
        # Count English words (less token-dense)
        english_pattern = r'\b[a-zA-Z]+\b'
        english_words = len(re.findall(english_pattern, text))
        
        # Count punctuation and special characters
        punctuation_pattern = r'[^\w\s\u4e00-\u9fff]'
        punctuation_count = len(re.findall(punctuation_pattern, text))
        
        # Count remaining characters (numbers, spaces, etc.)
        other_chars = len(text) - chinese_chars - english_words - punctuation_count
        
        # Estimation formula based on token density patterns:
        # - Chinese characters: ~1.5 chars per token
        # - English words: ~1.3 words per token  
        # - Punctuation: ~2 chars per token
        # - Other chars: ~3 chars per token
        
        chinese_tokens = chinese_chars / 1.5
        english_tokens = english_words / 1.3
        punctuation_tokens = punctuation_count / 2.0
        other_tokens = other_chars / 3.0
        
        total_estimated = chinese_tokens + english_tokens + punctuation_tokens + other_tokens
        
        # Add 15% buffer for safety
        return max(1, int(total_estimated * 1.15))


class LLMEngineRegistry:
    """Registry for LLM engines."""
    
    def __init__(self):
        self._engines: Dict[str, type] = {}
        self._import_errors: Dict[str, Exception] = {}
        self._preloaded = False
        # Remove preloading from __init__ to avoid circular imports
    
    def register(self, name: str, engine_class: type) -> None:
        """Register an LLM engine.
        
        Args:
            name: Engine name (e.g., 'openai', 'ollama')
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
        self._ensure_preloaded()
        if name not in self._engines:
            available = ", ".join(self.list_engines())
            import_errors = self.get_import_errors()
            
            error_msg = f"Unknown LLM engine: {name}."
            if available:
                error_msg += f" Available engines: {available}."
            if name in import_errors:
                error_msg += f" Note: '{name}' failed to load due to: {import_errors[name]}"
            
            raise KeyError(error_msg)
        
        return self._engines[name]
    
    def list_engines(self) -> List[str]:
        """Get list of registered engine names."""
        self._ensure_preloaded()
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
    
    def _ensure_preloaded(self) -> None:
        """Ensure all LLM providers have been preloaded."""
        if not self._preloaded:
            self._preload_engines()
    
    def _preload_engines(self) -> None:
        """主动导入所有LLM提供商模块."""
        if self._preloaded:
            return
        
        # 延迟导入提供商模块来避免循环导入
        # 这些导入会触发各提供商的注册代码
        import importlib
        
        known_providers = {
            'openai': 'nice_tts.engines.llm.openai_provider',
            'ollama': 'nice_tts.engines.llm.ollama_provider'
        }
        
        for provider_name, module_path in known_providers.items():
            try:
                # 导入模块，这会触发模块末尾的注册代码
                importlib.import_module(module_path)
            except Exception as e:
                # 记录导入失败的提供商
                self._import_errors[provider_name] = e
        
        self._preloaded = True
    
    def get_import_errors(self) -> Dict[str, Exception]:
        """获取导入失败的提供商信息.
        
        Returns:
            Dict[str, Exception]: 提供商名称到异常的映射
        """
        self._ensure_preloaded()
        return self._import_errors.copy()
    
    def get_debug_info(self) -> Dict[str, Any]:
        """获取注册表调试信息.
        
        Returns:
            Dict[str, Any]: 调试信息
        """
        self._ensure_preloaded()
        return {
            "available_engines": list(self._engines.keys()),
            "import_errors": {k: str(v) for k, v in self._import_errors.items()},
            "preloaded": self._preloaded,
            "total_engines": len(self._engines)
        }


# Global registry instance (using delayed initialization)
_registry_instance: Optional[LLMEngineRegistry] = None

def get_registry() -> LLMEngineRegistry:
    """Get LLM engine registry instance (singleton pattern).
    
    This function implements lazy initialization to avoid circular import issues.
    The registry is created only when first accessed, and providers are loaded
    on-demand rather than during module import.
    
    Returns:
        LLMEngineRegistry: The global registry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = LLMEngineRegistry()
    return _registry_instance

# For backward compatibility: provide registry as a property
class _RegistryProperty:
    @property
    def __get__(self, obj, objtype=None):
        return get_registry()

# Create registry instance for backward compatibility
registry = get_registry()