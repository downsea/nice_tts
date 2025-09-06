"""Ollama LLM provider implementation.

This module provides an Ollama-based LLM engine that works with
local Ollama installations, supporting models like llama2, mistral, etc.
"""

import json
import time
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None

from .base import LLMEngine, RefinementResult, SummaryResult, get_registry
from ...core.config import LLMConfig
from ...core.exceptions import (
    APIError, APIAuthenticationError, APIRateLimitError, APIQuotaExceededError,
    TokenLimitError, NetworkError, TimeoutError as NiceTTSTimeoutError
)


class OllamaProvider(LLMEngine):
    """Ollama LLM provider for local language models.
    
    Supports two modes:
    1. Native Ollama API (default) - uses /api/generate endpoint
    2. OpenAI-compatible API - uses /v1/chat/completions endpoint
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize Ollama provider.
        
        Args:
            config: LLM configuration
        """
        super().__init__(config)
        self._validate_config()
        self._determine_api_mode()
        self._setup_client()
    
    def refine_text(self, text: str) -> RefinementResult:
        """Refine transcribed text using Ollama LLM.
        
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
        
        # Split text into manageable chunks
        chunks = self.chunk_text(text, max_tokens=self.config.max_tokens - 2000)  # Reserve tokens for prompt
        
        refined_chunks = []
        total_tokens = 0
        
        for i, chunk in enumerate(chunks):
            try:
                refined_chunk, tokens_used = self._refine_chunk(chunk, i + 1, len(chunks))
                refined_chunks.append(refined_chunk)
                total_tokens += tokens_used
                
            except Exception as e:
                processing_time = time.time() - start_time
                raise self._handle_api_error(e, context={
                    "operation": "refine_text",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "processing_time": processing_time
                })
        
        processing_time = time.time() - start_time
        refined_text = "\\n\\n".join(refined_chunks)
        
        return RefinementResult(
            refined_text=refined_text,
            chunks_processed=len(chunks),
            tokens_used=total_tokens,
            processing_time=processing_time,
            metadata={
                "model": self.config.model_name,
                "provider": "ollama",
                "temperature": self.config.temperature
            }
        )
    
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
                    "provider": "ollama",
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
        """Count tokens in text (rough estimation for Ollama models).
        
        Since Ollama doesn't provide a direct tokenization endpoint,
        we use a rough estimation based on character count.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            int: Estimated number of tokens
        """
        # Different models have different tokenization patterns
        # This is a rough estimation: 1 token ≈ 3-4 characters for most models
        # For Chinese text, it might be closer to 1 token ≈ 2-3 characters
        
        char_count = len(text)
        
        # Adjust estimation based on content type
        chinese_chars = len([c for c in text if ord(c) > 127])
        english_chars = char_count - chinese_chars
        
        # Chinese characters tend to be more dense (fewer chars per token)
        chinese_tokens = chinese_chars // 2
        english_tokens = english_chars // 4
        
        return max(1, chinese_tokens + english_tokens)
    
    def get_max_tokens(self) -> int:
        """Get maximum token limit."""
        return self.config.max_tokens
    
    def validate_connection(self) -> bool:
        """Validate Ollama service connection."""
        try:
            if self.use_openai_compatible:
                return self._validate_openai_compatible_connection()
            else:
                return self._validate_native_connection()
                
        except Exception as e:
            raise self._handle_api_error(e, context={"operation": "validate_connection"})
    
    def _validate_native_connection(self) -> bool:
        """Validate native Ollama API connection."""
        # Test if Ollama service is running
        response = requests.get(f"{self.base_url}/api/tags", timeout=5)
        if response.status_code != 200:
            raise APIError(f"Ollama service returned status {response.status_code}")
        
        # Check if the specified model is available
        models = response.json().get("models", [])
        model_names = [model.get("name", "") for model in models]
        
        if self.config.model_name not in model_names:
            # Try to pull the model if it's not available
            return self._pull_model_if_needed()
        
        # Test generation with the model
        test_response = self._make_request("/api/generate", {
            "model": self.config.model_name,
            "prompt": "Hello",
            "stream": False
        })
        
        return test_response.get("response") is not None
    
    def _validate_openai_compatible_connection(self) -> bool:
        """Validate OpenAI-compatible API connection."""
        try:
            # Test with a simple chat completion
            response = self._openai_client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                temperature=0
            )
            return response.choices[0].message.content is not None
            
        except Exception as e:
            raise self._handle_openai_error(e)
    
    def _validate_config(self) -> None:
        """Validate configuration for Ollama provider."""
        # Ollama doesn't require API key, but we need base_url and model_name
        if not self.config.base_url:
            # Default to localhost if not specified
            self.config.base_url = "http://localhost:11434"
        
        if not self.config.model_name:
            raise ValueError("Ollama model name is required")
        
        self.base_url = self.config.base_url.rstrip("/")
    
    def _determine_api_mode(self) -> None:
        """Determine whether to use native Ollama API or OpenAI-compatible API.
        
        OpenAI-compatible mode is used if:
        1. base_url contains '/v1' path
        2. api_key is provided (indicating OpenAI-style authentication)
        3. OpenAI library is available
        """
        self.use_openai_compatible = (
            HAS_OPENAI and 
            ("/v1" in self.config.base_url or self.config.api_key)
        )
        
        if self.use_openai_compatible:
            # Ensure base_url ends with /v1 for OpenAI compatibility
            if not self.config.base_url.endswith("/v1"):
                if "/v1" not in self.config.base_url:
                    self.config.base_url = self.config.base_url.rstrip("/") + "/v1"
        
        self.api_mode = "openai-compatible" if self.use_openai_compatible else "native"
    
    def _setup_client(self) -> None:
        """Setup client for Ollama API."""
        if self.use_openai_compatible:
            # Use OpenAI client for compatibility mode
            self._openai_client = OpenAI(
                api_key=self.config.api_key or "ollama",  # Some implementations require a dummy key
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
        else:
            # Use requests session for native Ollama API
            self.session = requests.Session()
            self.session.timeout = self.config.timeout
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to Ollama API.
        
        Args:
            endpoint: API endpoint (e.g., "/api/generate")
            data: Request payload
            
        Returns:
            Dict[str, Any]: Response data
            
        Raises:
            APIError: If request fails
        """
        url = self.base_url + endpoint
        
        try:
            response = self.session.post(
                url,
                json=data,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.Timeout:
            raise NiceTTSTimeoutError(f"Ollama request timed out after {self.config.timeout}s")
        except requests.exceptions.ConnectionError:
            raise NetworkError(f"Cannot connect to Ollama at {self.base_url}")
        except requests.exceptions.HTTPError as e:
            raise APIError(f"Ollama API error: {e}")
        except json.JSONDecodeError:
            raise APIError("Invalid JSON response from Ollama")
    
    def _refine_chunk(self, chunk: str, chunk_num: int, total_chunks: int) -> tuple[str, int]:
        """Refine a single chunk of text.
        
        Args:
            chunk: Text chunk to refine
            chunk_num: Current chunk number (1-based)
            total_chunks: Total number of chunks
            
        Returns:
            tuple[str, int]: (refined_text, estimated_tokens_used)
        """
        # Prepare refinement prompt optimized for Chinese content
        prompt = self._get_refinement_prompt(chunk, chunk_num, total_chunks)
        
        if self.use_openai_compatible:
            return self._refine_chunk_openai_compatible(prompt)
        else:
            return self._refine_chunk_native(prompt)
    
    def _refine_chunk_native(self, prompt: str) -> tuple[str, int]:
        """Refine chunk using native Ollama API."""
        request_data = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        try:
            response = self._make_request("/api/generate", request_data)
            refined_text = response.get("response", "").strip()
            
            # Estimate tokens used (prompt + response)
            tokens_used = self.count_tokens(prompt) + self.count_tokens(refined_text)
            
            return refined_text, tokens_used
        except Exception as e:
            # Check if it's a token limit error
            if "maximum context length" in str(e).lower() or "token" in str(e).lower():
                # Try with smaller chunk
                smaller_chunks = self._emergency_split_chunk(prompt)
                refined_chunks = []
                total_tokens = 0
                
                for sub_chunk in smaller_chunks:
                    try:
                        sub_response = self._make_request("/api/generate", {
                            "model": self.config.model_name,
                            "prompt": sub_chunk,
                            "stream": False,
                            "options": {
                                "temperature": self.config.temperature,
                                "top_p": 0.9,
                                "top_k": 40
                            }
                        })
                        sub_refined = sub_response.get("response", "").strip()
                        refined_chunks.append(sub_refined)
                        total_tokens += self.count_tokens(sub_chunk) + self.count_tokens(sub_refined)
                    except Exception:
                        # Final fallback: return chunk as-is with warning
                        refined_chunks.append(f"[处理失败，保留原文]: {sub_chunk}")
                
                return "\n\n".join(refined_chunks), total_tokens
            else:
                raise self._handle_api_error(e)
    
    def _refine_chunk_openai_compatible(self, prompt: str) -> tuple[str, int]:
        """Refine chunk using OpenAI-compatible API."""
        try:
            response = self._openai_client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens // 2,  # Reserve space for prompt
                top_p=0.9
            )
            
            refined_text = response.choices[0].message.content.strip()
            
            # Get actual token usage if available
            if hasattr(response, 'usage') and response.usage:
                tokens_used = response.usage.total_tokens
            else:
                # Estimate tokens used
                tokens_used = self.count_tokens(prompt) + self.count_tokens(refined_text)
            
            return refined_text, tokens_used
            
        except Exception as e:
            # Check if it's a token limit error
            if "maximum context length" in str(e).lower() or "token" in str(e).lower():
                # Try with smaller chunk
                smaller_chunks = self._emergency_split_chunk(prompt)
                refined_chunks = []
                total_tokens = 0
                
                for sub_chunk in smaller_chunks:
                    try:
                        sub_response = self._openai_client.chat.completions.create(
                            model=self.config.model_name,
                            messages=[
                                {"role": "user", "content": sub_chunk}
                            ],
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens // 4,  # Reserve space for prompt
                            top_p=0.9
                        )
                        sub_refined = sub_response.choices[0].message.content.strip()
                        refined_chunks.append(sub_refined)
                        
                        # Get actual token usage if available
                        if hasattr(sub_response, 'usage') and sub_response.usage:
                            total_tokens += sub_response.usage.total_tokens
                        else:
                            # Estimate tokens used
                            total_tokens += self.count_tokens(sub_chunk) + self.count_tokens(sub_refined)
                    except Exception:
                        # Final fallback: return chunk as-is with warning
                        refined_chunks.append(f"[处理失败，保留原文]: {sub_chunk}")
                
                return "\n\n".join(refined_chunks), total_tokens
            else:
                raise self._handle_openai_error(e)
    
    def _summarize_single_chunk(
        self, 
        text: str, 
        audio_path: Optional[str] = None,
        original_txt_path: Optional[str] = None,
        refined_txt_path: Optional[str] = None
    ) -> tuple[str, int]:
        """Summarize a single chunk of text.
        
        Args:
            text: Text to summarize
            audio_path: Path to audio file (for metadata)
            original_txt_path: Path to original transcript
            refined_txt_path: Path to refined transcript
            
        Returns:
            tuple[str, int]: (summary_markdown, estimated_tokens_used)
        """
        prompt = self._get_summary_prompt(text, audio_path, original_txt_path, refined_txt_path)
        
        if self.use_openai_compatible:
            return self._summarize_openai_compatible(prompt)
        else:
            return self._summarize_native(prompt)
    
    def _summarize_native(self, prompt: str) -> tuple[str, int]:
        """Summarize using native Ollama API."""
        request_data = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        response = self._make_request("/api/generate", request_data)
        summary_md = response.get("response", "").strip()
        
        # Estimate tokens used
        tokens_used = self.count_tokens(prompt) + self.count_tokens(summary_md)
        
        return summary_md, tokens_used
    
    def _summarize_openai_compatible(self, prompt: str) -> tuple[str, int]:
        """Summarize using OpenAI-compatible API."""
        try:
            response = self._openai_client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens // 2,
                top_p=0.9
            )
            
            summary_md = response.choices[0].message.content.strip()
            
            # Get actual token usage if available
            if hasattr(response, 'usage') and response.usage:
                tokens_used = response.usage.total_tokens
            else:
                # Estimate tokens used
                tokens_used = self.count_tokens(prompt) + self.count_tokens(summary_md)
            
            return summary_md, tokens_used
            
        except Exception as e:
            raise self._handle_openai_error(e)
    
    def _summarize_multi_chunk(
        self,
        chunks: List[str],
        audio_path: Optional[str] = None,
        original_txt_path: Optional[str] = None,
        refined_txt_path: Optional[str] = None
    ) -> tuple[str, int]:
        """Summarize multiple chunks of text.
        
        Args:
            chunks: List of text chunks to summarize
            audio_path: Path to audio file
            original_txt_path: Path to original transcript
            refined_txt_path: Path to refined transcript
            
        Returns:
            tuple[str, int]: (summary_markdown, total_tokens_used)
        """
        # First, create individual summaries for each chunk
        chunk_summaries = []
        total_tokens = 0
        
        for i, chunk in enumerate(chunks):
            prompt = self._get_chunk_summary_prompt(chunk, i + 1, len(chunks))
            
            request_data = {
                "model": self.config.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            response = self._make_request("/api/generate", request_data)
            chunk_summary = response.get("response", "").strip()
            chunk_summaries.append(chunk_summary)
            
            total_tokens += self.count_tokens(prompt) + self.count_tokens(chunk_summary)
        
        # Then combine all chunk summaries into a final summary
        combined_summary_text = "\\n\\n".join(chunk_summaries)
        final_prompt = self._get_final_summary_prompt(
            combined_summary_text, audio_path, original_txt_path, refined_txt_path
        )
        
        request_data = {
            "model": self.config.model_name,
            "prompt": final_prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        response = self._make_request("/api/generate", request_data)
        final_summary = response.get("response", "").strip()
        
        total_tokens += self.count_tokens(final_prompt) + self.count_tokens(final_summary)
        
        return final_summary, total_tokens
    
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
    
    def _get_refinement_prompt(self, chunk: str, chunk_num: int, total_chunks: int) -> str:
        """Generate refinement prompt for text chunk."""
        context_info = ""
        if total_chunks > 1:
            context_info = f"（这是第{chunk_num}部分，共{total_chunks}部分）"
        
        return f"""请对以下音频转录文本进行语言修正和优化{context_info}：

原始转录文本：
{chunk}

请按以下要求处理：
1. 修正明显的转录错误和语法问题
2. 补充合理的标点符号
3. 调整语句结构使其更加流畅自然
4. 保持原意不变，不要添加原文没有的内容
5. 输出修正后的文本，不要包含解释说明

修正后的文本："""
    
    def _get_summary_prompt(
        self, 
        text: str, 
        audio_path: Optional[str] = None,
        original_txt_path: Optional[str] = None,
        refined_txt_path: Optional[str] = None
    ) -> str:
        """Generate summary prompt for text."""
        
        # Prepare file links
        file_links = []
        if audio_path:
            file_links.append(f"- [原始音频]({Path(audio_path).name})")
        if original_txt_path:
            file_links.append(f"- [原始转录]({Path(original_txt_path).name})")
        if refined_txt_path:
            file_links.append(f"- [精校文本]({Path(refined_txt_path).name})")
        
        file_section = ""
        if file_links:
            file_section = f"""

## 相关文件
{chr(10).join(file_links)}"""
        
        return f"""请为以下会议内容生成结构化的Markdown格式摘要：

会议内容：
{text}

请按以下格式输出摘要：

# 会议纪要

## 会议详情
- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}
- 参与者：[根据内容识别]

## 纪要概述
[用2-3句话概括会议主要内容]

## 主要讨论点
[列出关键讨论议题，使用项目符号]

## 行动项
[列出具体的行动计划和负责人]

## 关键决策
[列出会议中做出的重要决定]
{file_section}

请确保摘要准确反映会议内容，使用清晰的中文表达："""
    
    def _get_chunk_summary_prompt(self, chunk: str, chunk_num: int, total_chunks: int) -> str:
        """Generate prompt for summarizing individual chunks."""
        return f"""请为以下会议内容片段生成简洁摘要（第{chunk_num}部分，共{total_chunks}部分）：

内容片段：
{chunk}

请提取本片段的关键信息：
1. 主要讨论话题
2. 重要观点或决定
3. 行动项（如有）

简洁摘要："""
    
    def _get_final_summary_prompt(
        self,
        combined_summaries: str,
        audio_path: Optional[str] = None,
        original_txt_path: Optional[str] = None,
        refined_txt_path: Optional[str] = None
    ) -> str:
        """Generate prompt for final combined summary."""
        
        # Prepare file links
        file_links = []
        if audio_path:
            file_links.append(f"- [原始音频]({Path(audio_path).name})")
        if original_txt_path:
            file_links.append(f"- [原始转录]({Path(original_txt_path).name})")
        if refined_txt_path:
            file_links.append(f"- [精校文本]({Path(refined_txt_path).name})")
        
        file_section = ""
        if file_links:
            file_section = f"""

## 相关文件
{chr(10).join(file_links)}"""
        
        return f"""基于以下各部分的摘要，请生成完整的会议纪要：

分段摘要：
{combined_summaries}

请整合所有信息，生成完整的Markdown格式会议纪要：

# 会议纪要

## 会议详情
- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}
- 参与者：[根据内容识别]

## 纪要概述
[整合所有部分，用2-3句话概括整个会议]

## 主要讨论点
[合并所有关键讨论议题]

## 行动项
[汇总所有行动计划]

## 关键决策
[汇总所有重要决定]
{file_section}

完整纪要："""
    
    def _handle_api_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Exception:
        """Handle and convert API errors to nice-tts exceptions."""
        if context is None:
            context = {}
        
        context["api_mode"] = self.api_mode
        
        if isinstance(error, requests.exceptions.ConnectionError):
            return NetworkError(
                "Cannot connect to Ollama service. Make sure Ollama is running.",
                details={**context, "suggestion": "Start Ollama with 'ollama serve'"}
            )
        elif isinstance(error, requests.exceptions.Timeout):
            return NiceTTSTimeoutError(
                "Ollama request timed out",
                details=context
            )
        elif isinstance(error, requests.exceptions.HTTPError):
            return APIError(
                f"Ollama API error: {error}",
                provider="ollama",
                details=context
            )
        else:
            return APIError(
                f"Unexpected Ollama error: {error}",
                provider="ollama",
                details={**context, "error_type": type(error).__name__}
            )
    
    def _handle_openai_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Exception:
        """Handle OpenAI-compatible API errors."""
        if context is None:
            context = {}
        
        context["api_mode"] = "openai-compatible"
        
        try:
            from openai import APIError as OpenAIAPIError, RateLimitError, AuthenticationError
            
            if isinstance(error, AuthenticationError):
                return APIAuthenticationError(
                    "Ollama OpenAI-compatible API authentication failed",
                    provider="ollama",
                    details=context
                )
            elif isinstance(error, RateLimitError):
                return APIRateLimitError(
                    "Ollama API rate limit exceeded",
                    provider="ollama",
                    details=context
                )
            elif isinstance(error, OpenAIAPIError):
                return APIError(
                    f"Ollama OpenAI-compatible API error: {error}",
                    provider="ollama",
                    details=context
                )
            else:
                return APIError(
                    f"Unexpected Ollama OpenAI-compatible error: {error}",
                    provider="ollama",
                    details={**context, "error_type": type(error).__name__}
                )
        except ImportError:
            # Fallback if OpenAI library not available
            return APIError(
                f"Ollama error: {error}",
                provider="ollama",
                details={**context, "error_type": type(error).__name__}
            )


# Register the Ollama provider
get_registry().register("ollama", OllamaProvider)