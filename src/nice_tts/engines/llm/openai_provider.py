"""OpenAI LLM provider implementation.

This module provides an OpenAI-compatible LLM engine that works with
OpenAI's API as well as compatible services.
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from openai import OpenAI
from transformers import AutoTokenizer

from .base import LLMEngine, RefinementResult, SummaryResult, get_registry
from ...core.config import LLMConfig
from ...core.exceptions import (
    APIError, APIAuthenticationError, APIRateLimitError, APIQuotaExceededError,
    TokenLimitError, TokenizerError, ChunkingError, NetworkError, TimeoutError as NiceTTSTimeoutError
)


class OpenAIProvider(LLMEngine):
    """OpenAI-compatible LLM provider."""
    
    def __init__(self, config: LLMConfig):
        """Initialize OpenAI provider.
        
        Args:
            config: LLM configuration
        """
        super().__init__(config)
        self._validate_config()
        self._setup_client()
        self._setup_tokenizer()
    
    def refine_text(self, text: str) -> RefinementResult:
        """Refine transcribed text using OpenAI LLM.
        
        Args:
            text: Raw transcribed text to refine
            
        Returns:
            RefinementResult: The refinement result
        """
        if not text.strip():
            return RefinementResult(
                refined_text="",
                chunks_processed=0,
                tokens_used=0,
                processing_time=0.0,
                metadata={"reason": "empty_input"}
            )
        
        start_time = time.time()
        
        try:
            # Use improved chunking with safety checks
            chunks = self._safe_chunk_text(text, reserve_tokens=2000)  # Reserve for prompt
            
            refined_chunks = []
            total_tokens = 0
            
            for i, chunk in enumerate(chunks):
                try:
                    refined_chunk, tokens_used = self._refine_chunk(chunk, i + 1, len(chunks))
                    refined_chunks.append(refined_chunk)
                    total_tokens += tokens_used
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    
                    # Check if it's a token limit error
                    if "maximum sequence length" in str(e).lower() or "token" in str(e).lower():
                        # Try with smaller chunk
                        smaller_chunks = self._emergency_split_chunk(chunk)
                        for sub_chunk in smaller_chunks:
                            try:
                                refined_sub_chunk, sub_tokens = self._refine_chunk(
                                    sub_chunk, i + 1, len(chunks), emergency=True
                                )
                                refined_chunks.append(refined_sub_chunk)
                                total_tokens += sub_tokens
                            except Exception as sub_e:
                                # Final fallback: return chunk as-is with warning
                                refined_chunks.append(f"[处理失败，保留原文]: {sub_chunk}")
                    else:
                        raise self._handle_api_error(e, context={
                            "operation": "refine_text",
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "processing_time": processing_time
                        })
            
            processing_time = time.time() - start_time
            refined_text = "\n\n".join(refined_chunks)
            
            return RefinementResult(
                refined_text=refined_text,
                chunks_processed=len(chunks),
                tokens_used=total_tokens,
                processing_time=processing_time,
                metadata={
                    "model": self.config.model_name,
                    "provider": "openai",
                    "temperature": self.config.temperature
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            raise self._handle_api_error(e, context={
                "operation": "refine_text",
                "processing_time": processing_time
            })
    
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
            audio_path: Path to original audio file
            original_txt_path: Path to original transcript
            refined_txt_path: Path to refined transcript
            
        Returns:
            SummaryResult: The summary result
        """
        if not text.strip():
            return SummaryResult(
                summary_markdown="",
                tokens_used=0,
                processing_time=0.0,
                metadata={"reason": "empty_input"}
            )
        
        start_time = time.time()
        
        try:
            # Check if text needs to be chunked for summarization
            chunks = self.chunk_text(text, max_tokens=self.config.max_tokens - 3000)  # Reserve for prompt
            
            if len(chunks) > 1:
                # Multi-chunk summarization
                summary_md, tokens_used = self._summarize_multi_chunk(
                    chunks, audio_path, original_txt_path, refined_txt_path
                )
            else:
                # Single chunk summarization
                summary_md, tokens_used = self._summarize_single_chunk(
                    text, audio_path, original_txt_path, refined_txt_path
                )
            
            processing_time = time.time() - start_time
            
            return SummaryResult(
                summary_markdown=summary_md,
                tokens_used=tokens_used,
                processing_time=processing_time,
                metadata={
                    "model": self.config.model_name,
                    "provider": "openai",
                    "chunks_processed": len(chunks),
                    "audio_path": audio_path
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            raise self._handle_api_error(e, context={
                "operation": "summarize_text",
                "processing_time": processing_time,
                "audio_path": audio_path
            })
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the configured tokenizer."""
        if not self._tokenizer:
            self._setup_tokenizer()
        
        # For safety, limit text length to prevent tokenizer errors
        # If text is extremely long, use character-based estimation
        if len(text) > 50000:  # 50K characters threshold
            return self._estimate_tokens_by_chars(text)
        
        try:
            # Ensure tokenizer can handle the text length
            if hasattr(self._tokenizer, 'model_max_length'):
                max_length = getattr(self._tokenizer, 'model_max_length', 512)
                if max_length and max_length < 2048:  # Avoid small limits like GPT-2's 1024
                    return self._estimate_tokens_by_chars(text)
                    
            return len(self._tokenizer.encode(text, add_special_tokens=False, truncation=False))
        except Exception as e:
            # Handle tokenizer-specific errors
            if "maximum sequence length" in str(e).lower() or "exceeds the maximum" in str(e).lower():
                # This is a tokenizer length limit error
                raise TokenLimitError.from_tokenizer_error(e, len(text), 1024)
            elif "model" in str(e).lower() and "not found" in str(e).lower():
                # Tokenizer model loading error
                raise TokenizerError(
                    f"Failed to load tokenizer model: {e}",
                    model_name=self.config.model_name,
                    details={"error_type": type(e).__name__, "original_error": str(e)}
                )
            else:
                # Fallback: accurate estimation based on text characteristics
                return self._estimate_tokens_by_chars(text)
    
    def get_max_tokens(self) -> int:
        """Get maximum token limit."""
        return self.config.max_tokens
    
    def validate_connection(self) -> bool:
        """Validate OpenAI API connection."""
        try:
            # Make a minimal API call to test connection
            response = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1,
                temperature=0
            )
            return True
            
        except Exception as e:
            raise self._handle_api_error(e, context={"operation": "validate_connection"})
    
    def _validate_config(self) -> None:
        """Validate configuration for OpenAI provider."""
        if not self.config.api_key:
            raise APIAuthenticationError("OpenAI API key is required")
        
        if not self.config.base_url:
            raise ValueError("OpenAI base URL is required")
        
        if not self.config.model_name:
            raise ValueError("OpenAI model name is required")
    
    def _setup_client(self) -> None:
        """Setup OpenAI client."""
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout
        )
    
    def _setup_tokenizer(self) -> None:
        """Setup tokenizer for token counting."""
        try:
            # Try to load tokenizer for the specific model
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        except Exception:
            try:
                # Use cl100k_base tokenizer for OpenAI models (GPT-4, GPT-3.5-turbo)
                # This tokenizer supports much longer sequences and is more accurate
                self._tokenizer = AutoTokenizer.from_pretrained("Xenova/gpt-4")
            except Exception:
                try:
                    # Second fallback: Use a more modern tokenizer
                    self._tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
                except Exception:
                    # Final fallback: None (will use character-based estimation)
                    self._tokenizer = None
    
    def _refine_chunk(self, chunk: str, chunk_num: int, total_chunks: int, emergency: bool = False) -> tuple[str, int]:
        """Refine a single text chunk.
        
        Args:
            chunk: Text chunk to refine
            chunk_num: Current chunk number (1-based)
            total_chunks: Total number of chunks
            emergency: Whether this is an emergency retry with smaller limits
            
        Returns:
            Tuple of (refined_text, tokens_used)
        """
        system_prompt = "你是一位专业的速记稿后期处理专家，请使用中文输出。"
        
        base_prompt = f"""您是一位专业的速记稿后期处理专家。您的任务是优化下方由自动语音识别（ASR）生成的原始会议记录文本。请严格按照以下要求操作，以生成一份流畅、准确、易读的精校版中文文稿。

**处理要求：**
1.  **润色和修正**：修正文本中的错别字、语法错误，并确保语句通顺、表达准确。
2.  **优化标点符号**：根据上下文和語氣，正确使用中文标点符号，如逗号（，）、句号（。）、顿号（、）、问号（？）等。
3.  **去除冗余**：删除对话中的无意义口头禅、重复词语和语气词（例如："嗯"、"呃"、"那个"、"就是说"等）。
4.  **合并与分段**：将碎片化句子合并为完整、连贯的句子。根据内容的逻辑关系，将文本合理地划分为段落。
5.  **保持原意**：必须最大限度地保留原始对话的意图和信息。
6.  **格式要求**：最终输出结果应为纯净的、连续的中文文本段落。**不要**添加任何标题、说话人标识或任何形式的标记。"""
        
        # Adjust prompt for emergency mode
        if emergency:
            base_prompt += "\n\n**特别注意**：请尽可能简洁地处理，保持核心内容。"
        
        chunk_info = f'\n这是第 {chunk_num}/{total_chunks} 部分文本。' if total_chunks > 1 else ''
        
        user_prompt = f"""{base_prompt}{chunk_info}

**原始速记稿文本如下：**
---
{chunk}
---"""
        
        # Adjust token limits for emergency mode
        max_output_tokens = self.config.max_tokens // 4 if emergency else self.config.max_tokens // 2
        
        response = self._client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=max_output_tokens
        )
        
        refined_text = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens if response.usage else 0
        
        return refined_text, tokens_used
    
    def _estimate_tokens_by_chars(self, text: str) -> int:
        """Estimate tokens based on text characteristics.
        
        This method provides more accurate token estimation than simple
        character division, taking into account language and content type.
        """
        if not text:
            return 0
            
        # Count different types of characters
        chinese_chars = 0
        english_words = 0
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
        
        # Estimation formula based on token density patterns:
        # - Chinese characters: ~1.5 chars per token
        # - English words: ~1.3 words per token  
        # - Punctuation: ~2 chars per token
        
        chinese_tokens = chinese_chars / 1.5
        english_tokens = english_words / 1.3
        punctuation_tokens = punctuation_count / 2.0
        
        total_estimated = chinese_tokens + english_tokens + punctuation_tokens
        
        # Add 10% buffer for safety
        return max(1, int(total_estimated * 1.1))
    
    def _safe_chunk_text(self, text: str, reserve_tokens: int = 2000) -> List[str]:
        """Safely chunk text with error handling and fallback strategies.
        
        Args:
            text: Text to chunk
            reserve_tokens: Tokens to reserve for prompt and output
            
        Returns:
            List[str]: Safely chunked text
            
        Raises:
            ChunkingError: If chunking fails completely
        """
        try:
            # First, try normal chunking
            max_chunk_tokens = self.config.max_tokens - reserve_tokens
            chunks = self.chunk_text(text, max_tokens=max_chunk_tokens)
            
            # Validate each chunk
            safe_chunks = []
            for i, chunk in enumerate(chunks):
                try:
                    chunk_tokens = self.count_tokens(chunk)
                    if chunk_tokens > max_chunk_tokens:
                        # Split oversized chunk further
                        sub_chunks = self._emergency_split_chunk(chunk, max_chunk_tokens)
                        safe_chunks.extend(sub_chunks)
                    else:
                        safe_chunks.append(chunk)
                except (TokenLimitError, TokenizerError) as e:
                    # Handle token-related errors gracefully
                    sub_chunks = self._emergency_split_chunk(chunk, max_chunk_tokens // 2)
                    safe_chunks.extend(sub_chunks)
            
            if not safe_chunks:
                raise ChunkingError(
                    "All chunks were empty after processing",
                    text_length=len(text),
                    max_tokens=max_chunk_tokens
                )
            
            return safe_chunks
            
        except (TokenLimitError, TokenizerError, ChunkingError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Fallback to character-based chunking for unexpected errors
            try:
                char_limit = (self.config.max_tokens - reserve_tokens) * 3  # Conservative estimate
                fallback_chunks = self._fallback_char_split(text, char_limit)
                if fallback_chunks:
                    return fallback_chunks
            except Exception:
                pass  # Final fallback below
            
            # Final fallback: raise ChunkingError
            raise ChunkingError(
                f"Chunking failed completely: {e}",
                text_length=len(text),
                max_tokens=self.config.max_tokens - reserve_tokens,
                details={"original_error": str(e), "error_type": type(e).__name__}
            )
    
    def _emergency_split_chunk(self, chunk: str, max_tokens: Optional[int] = None) -> List[str]:
        """Emergency splitting for oversized chunks.
        
        Args:
            chunk: Chunk to split
            max_tokens: Maximum tokens per sub-chunk
            
        Returns:
            List[str]: List of smaller chunks
        """
        if max_tokens is None:
            max_tokens = self.config.max_tokens // 4  # Very conservative
        
        # Split by sentences first
        import re
        sentences = re.split(r'([.。!?！？]+)', chunk)
        
        sub_chunks = []
        current_sub_chunk = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            test_chunk = current_sub_chunk + sentence
            if self.count_tokens(test_chunk) <= max_tokens:
                current_sub_chunk = test_chunk
            else:
                if current_sub_chunk:
                    sub_chunks.append(current_sub_chunk)
                # Handle oversized single sentence
                if self.count_tokens(sentence) > max_tokens:
                    # Word-level split as last resort
                    words = sentence.split()
                    word_chunk = ""
                    for word in words:
                        test_word_chunk = word_chunk + " " + word if word_chunk else word
                        if self.count_tokens(test_word_chunk) <= max_tokens:
                            word_chunk = test_word_chunk
                        else:
                            if word_chunk:
                                sub_chunks.append(word_chunk)
                            word_chunk = word
                    current_sub_chunk = word_chunk
                else:
                    current_sub_chunk = sentence
        
        if current_sub_chunk:
            sub_chunks.append(current_sub_chunk)
        
        return sub_chunks if sub_chunks else [chunk[:1000]]  # Final fallback
    
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
            boundary_chars = ['。', '.', '!', '?', '！', '？']
            best_break = end
            
            for i in range(end - 100, start, -1):  # Look back up to 100 chars
                if text[i] in boundary_chars:
                    best_break = i + 1
                    break
            
            chunks.append(text[start:best_break])
            start = best_break
        
        return chunks
    
    def _summarize_single_chunk(
        self,
        text: str,
        audio_path: Optional[str],
        original_txt_path: Optional[str],
        refined_txt_path: Optional[str]
    ) -> tuple[str, int]:
        """Summarize a single text chunk."""
        
        # Prepare file links
        links_section = self._create_links_section(audio_path, original_txt_path, refined_txt_path)
        
        system_prompt = "你是一位专业的会议助理，请使用中文输出Markdown格式的会议纪要。"
        
        user_prompt = f"""您是一位专业的会议助理。请根据下方提供的会议文稿，生成一份完整的中文会议纪要。

**指令：**
1.  **提取元信息**:
    *   **会议主题**: 从文稿内容中提炼出核心议题作为纪要标题。
    *   **会议日期**: {datetime.now().strftime('%Y-%m-%d')}（今天日期）。
    *   **参会人员**: 尝试从文稿中识别出参会者的姓名。如果无法明确识别，请注明"参会人员未明确"。

2.  **构建纪要结构 (Markdown格式)**: 纪要必须包含以下几个部分：
    *   `## 会议详情`: 包括推断出的会议主题、日期和参会人员。
    *   `## 纪要概述`: 一段话，简明扼要地总结会议的核心目的和主要成果。
    *   `## 主要讨论点`: 使用项目符号（bullet points）列出会议中讨论的主要议题。
    *   `## 行动项`: 使用编号列表（numbered list）明确记录需要执行的任务。如果可能，请指明负责人。如果没有明确的行动项，请注明"无"。
    *   `## 关键决策`: 使用项目符号记录会议中达成的关键决定。如果没有，请注明"无"。

{links_section}

**优化后的会议文稿:**
---
{text}
---

请现在开始生成完整的中文Markdown格式会议纪要。"""
        
        response = self._client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=self.config.max_tokens // 2
        )
        
        summary_md = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens if response.usage else 0
        
        return summary_md, tokens_used
    
    def _summarize_multi_chunk(
        self,
        chunks: List[str],
        audio_path: Optional[str],
        original_txt_path: Optional[str],
        refined_txt_path: Optional[str]
    ) -> tuple[str, int]:
        """Summarize multiple text chunks."""
        
        # First, create summaries for each chunk
        chunk_summaries = []
        total_tokens = 0
        
        for i, chunk in enumerate(chunks):
            prompt = f"""您是一位专业的会议助理。请根据下方提供的会议文稿片段，生成一个该片段的简洁摘要。

这是第 {i + 1}/{len(chunks)} 部分文稿。

**会议文稿片段:**
---
{chunk}
---"""
            
            response = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "你是一位专业的会议助理，请使用中文输出摘要。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=self.config.max_tokens // 4
            )
            
            chunk_summary = response.choices[0].message.content or ""
            chunk_summaries.append(chunk_summary)
            total_tokens += response.usage.total_tokens if response.usage else 0
        
        # Combine chunk summaries and create final summary
        combined_summary = "\n\n---\n\n".join(chunk_summaries)
        
        links_section = self._create_links_section(audio_path, original_txt_path, refined_txt_path)
        
        final_prompt = f"""你是一位专业的会议助理，请根据下方提供的各部分摘要，整合成一份完整的、有条理的中文会议纪要。

**指令：**
1.  **提取元信息**:
    *   **会议主题**: 从摘要内容中提炼出核心议题作为纪要标题。
    *   **会议日期**: {datetime.now().strftime('%Y-%m-%d')}。
    *   **参会人员**: 尝试从摘要中识别出参会者。如果无法明确识别，请注明"参会人员未明确"。

2.  **构建纪要结构 (Markdown格式)**: 纪要必须包含以下几个部分：
    *   `## 会议详情`: 包括推断出的会议主题、日期和参会人员。
    *   `## 纪要概述`: 一段话，简明扼要地总结会议的核心目的和主要成果。
    *   `## 主要讨论点`: 使用项目符号列出会议中讨论的主要议题。
    *   `## 行动项`: 使用编号列表明确记录需要执行的任务。如果没有明确的行动项，请注明"无"。
    *   `## 关键决策`: 使用项目符号记录会议中达成的关键决定。如果没有，请注明"无"。

{links_section}

**各部分摘要:**
---
{combined_summary}
---

请现在开始生成完整的中文Markdown格式会议纪要。"""
        
        response = self._client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": "你是一位专业的会议助理，请使用中文输出Markdown格式的会议纪要。"},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.5,
            max_tokens=self.config.max_tokens // 2
        )
        
        final_summary = response.choices[0].message.content or ""
        total_tokens += response.usage.total_tokens if response.usage else 0
        
        return final_summary, total_tokens
    
    def _create_links_section(
        self,
        audio_path: Optional[str],
        original_txt_path: Optional[str],
        refined_txt_path: Optional[str]
    ) -> str:
        """Create the links section for the summary."""
        if not any([audio_path, original_txt_path, refined_txt_path]):
            return ""
        
        links = []
        if audio_path:
            audio_name = Path(audio_path).name
            links.append(f"- 原始录音: [{audio_name}](./{audio_name})")
        
        if original_txt_path:
            txt_name = Path(original_txt_path).name
            links.append(f"- 原始文本: [{txt_name}](./{txt_name})")
        
        if refined_txt_path:
            refined_name = Path(refined_txt_path).name
            links.append(f"- 精校文本: [{refined_name}](./{refined_name})")
        
        return f"""
3.  **包含相关文件链接**: 在纪要末尾，添加 `## 相关文件` 部分，并附上以下链接：
{chr(10).join(links)}
"""
    
    def _handle_api_error(self, error: Exception, context: Dict[str, Any]) -> Exception:
        """Convert API errors to nice-tts exceptions."""
        from openai import APIError as OpenAIAPIError, RateLimitError, AuthenticationError
        
        if isinstance(error, AuthenticationError):
            return APIAuthenticationError(
                "OpenAI API authentication failed",
                provider="openai",
                details=context
            )
        elif isinstance(error, RateLimitError):
            return APIRateLimitError(
                "OpenAI API rate limit exceeded",
                provider="openai",
                details=context
            )
        elif isinstance(error, OpenAIAPIError):
            return APIError(
                f"OpenAI API error: {error}",
                provider="openai",
                details=context
            )
        elif isinstance(error, TimeoutError):
            return NiceTTSTimeoutError(
                "OpenAI API request timed out",
                details=context
            )
        else:
            return APIError(
                f"Unexpected OpenAI error: {error}",
                provider="openai",
                details={**context, "error_type": type(error).__name__}
            )


# Register the OpenAI provider
get_registry().register("openai", OpenAIProvider)