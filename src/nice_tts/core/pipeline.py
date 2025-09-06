"""Processing pipeline for nice-tts.

This module implements the main processing pipeline that orchestrates
transcription, refinement, and summarization stages for audio files.
"""

import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from .config import AppConfig
from .exceptions import (
    PipelineError, ProcessingError, FatalError,
    is_retryable_error, get_retry_delay
)
from ..engines.transcription.base import registry as transcription_registry
from ..engines.llm.base import registry as llm_registry
from ..utils.file_manager import FileManager, ProcessingStage, FileInfo
from ..utils.logger import Logger, ProgressInfo, ProgressReporter


@dataclass
class StageResult:
    """Result of a processing stage."""
    
    stage: ProcessingStage
    success: bool
    output_path: Optional[Path] = None
    processing_time: float = 0.0
    error: Optional[Exception] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FileResult:
    """Result of processing a single file."""
    
    file_info: FileInfo
    success: bool
    stages_completed: List[StageResult]
    total_processing_time: float = 0.0
    error: Optional[Exception] = None


@dataclass  
class BatchResult:
    """Result of processing a batch of files."""
    
    files_processed: List[FileResult]
    total_files: int
    successful_files: int
    failed_files: int
    total_processing_time: float = 0.0
    errors: List[Exception] = field(default_factory=list)


class ProcessingPipeline:
    """Main processing pipeline for audio files."""
    
    def __init__(self, config: AppConfig):
        """Initialize processing pipeline.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.file_manager = FileManager(config.output)
        self.logger = Logger(config.logging)
        self.progress_reporter = ProgressReporter(self.logger)
        
        # Initialize engines
        self.transcription_engine = None
        self.llm_engine = None
        self._setup_engines()
    
    def process_batch(
        self,
        input_path: Path,
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None
    ) -> BatchResult:
        """Process a batch of audio files.
        
        Args:
            input_path: Path to audio file or directory
            progress_callback: Optional progress callback function
            
        Returns:
            BatchResult: Results of batch processing
        """
        start_time = time.time()
        
        try:
            # Discover audio files
            with self.logger.performance_timer("file_discovery", input_path=str(input_path)):
                audio_files = self.file_manager.discover_audio_files(input_path)
            
            self.logger.info(
                f"Found {len(audio_files)} audio files to process",
                files_found=len(audio_files),
                input_path=str(input_path)
            )
            
            # Create processing plan
            processing_plan = self.file_manager.get_processing_plan(
                audio_files, 
                force=self.config.output.force_reprocess
            )
            
            # Process files
            if self.config.parallel_jobs > 1:
                results = self._process_files_parallel(processing_plan, progress_callback)
            else:
                results = self._process_files_sequential(processing_plan, progress_callback)
            
            # Calculate statistics
            successful_files = sum(1 for r in results if r.success)
            failed_files = len(results) - successful_files
            errors = [r.error for r in results if r.error is not None]
            
            total_processing_time = time.time() - start_time
            
            # Log batch summary
            self.logger.log_batch_summary(
                successful_files,
                len(audio_files),
                total_processing_time
            )
            
            return BatchResult(
                files_processed=results,
                total_files=len(audio_files),
                successful_files=successful_files,
                failed_files=failed_files,
                total_processing_time=total_processing_time,
                errors=errors
            )
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}", error=str(e))
            raise PipelineError(f"Batch processing failed: {e}") from e
    
    def process_single_file(
        self,
        file_info: FileInfo,
        required_stages: List[ProcessingStage],
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None
    ) -> FileResult:
        """Process a single audio file.
        
        Args:
            file_info: File information
            required_stages: Stages that need processing
            progress_callback: Optional progress callback
            
        Returns:
            FileResult: Processing result
        """
        start_time = time.time()
        stage_results = []
        
        self.logger.log_file_processing(
            str(file_info.audio_path),
            "batch",
            "started",
            required_stages=[s.value for s in required_stages]
        )
        
        try:
            # Process each required stage
            for i, stage in enumerate(required_stages):
                
                # Update progress
                if progress_callback:
                    progress = ProgressInfo(
                        current=i,
                        total=len(required_stages),
                        stage=stage.value,
                        file_name=file_info.audio_path.name,
                        start_time=start_time
                    )
                    progress_callback(progress)
                
                # Process stage with retry logic
                stage_result = self._process_stage_with_retry(file_info, stage)
                stage_results.append(stage_result)
                
                # Stop on failure
                if not stage_result.success:
                    break
            
            # Final progress update
            if progress_callback:
                progress = ProgressInfo(
                    current=len(required_stages),
                    total=len(required_stages),
                    stage="completed",
                    file_name=file_info.audio_path.name,
                    start_time=start_time
                )
                progress_callback(progress)
            
            # Determine overall success
            success = all(r.success for r in stage_results)
            error = next((r.error for r in stage_results if r.error), None)
            
            total_time = time.time() - start_time
            
            # Log completion
            status = "completed" if success else "failed"
            self.logger.log_file_processing(
                str(file_info.audio_path),
                "batch",
                status,
                processing_time=total_time,
                stages_completed=len([r for r in stage_results if r.success])
            )
            
            return FileResult(
                file_info=file_info,
                success=success,
                stages_completed=stage_results,
                total_processing_time=total_time,
                error=error
            )
            
        except Exception as e:
            total_time = time.time() - start_time
            
            self.logger.log_file_processing(
                str(file_info.audio_path),
                "batch",
                "failed",
                processing_time=total_time,
                error=str(e)
            )
            
            return FileResult(
                file_info=file_info,
                success=False,
                stages_completed=stage_results,
                total_processing_time=total_time,
                error=e
            )
    
    def _setup_engines(self) -> None:
        """Setup transcription and LLM engines."""
        try:
            # Setup transcription engine
            self.transcription_engine = transcription_registry.create_engine(
                "whisper",  # Currently only Whisper is supported
                self.config.transcription
            )
            
            # Setup LLM engine
            self.llm_engine = llm_registry.create_engine(
                self.config.llm.provider,
                self.config.llm
            )
            
            self.logger.info(
                "Engines initialized successfully",
                transcription_engine="whisper",
                llm_engine=self.config.llm.provider
            )
            
        except Exception as e:
            self.logger.error(f"Failed to setup engines: {e}")
            raise FatalError(f"Engine initialization failed: {e}") from e
    
    def _process_files_sequential(
        self,
        processing_plan: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[ProgressInfo], None]]
    ) -> List[FileResult]:
        """Process files sequentially."""
        results = []
        
        for i, plan_item in enumerate(processing_plan):
            file_info = plan_item["file_info"]
            required_stages = plan_item["required_stages"]
            
            # Skip if no stages needed
            if not required_stages:
                self.logger.log_file_processing(
                    str(file_info.audio_path),
                    "all_stages",
                    "skipped",
                    reason="all_stages_complete"
                )
                
                result = FileResult(
                    file_info=file_info,
                    success=True,
                    stages_completed=[],
                    total_processing_time=0.0
                )
                results.append(result)
                continue
            
            # Create file-specific progress callback
            def file_progress_callback(stage_progress: ProgressInfo) -> None:
                # Combine file and stage progress
                overall_progress = ProgressInfo(
                    current=i * len(ProcessingStage) + stage_progress.current,
                    total=len(processing_plan) * len(ProcessingStage),
                    stage=stage_progress.stage,
                    file_name=stage_progress.file_name,
                    start_time=stage_progress.start_time
                )
                
                if progress_callback:
                    progress_callback(overall_progress)
                
                # Also report to progress reporter
                self.progress_reporter.report_progress(overall_progress)
            
            # Process file
            result = self.process_single_file(
                file_info,
                required_stages,
                file_progress_callback
            )
            results.append(result)
        
        return results
    
    def _process_files_parallel(
        self,
        processing_plan: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[ProgressInfo], None]]
    ) -> List[FileResult]:
        """Process files in parallel."""
        results = []
        completed_files = 0
        
        with ThreadPoolExecutor(max_workers=self.config.parallel_jobs) as executor:
            # Submit all jobs
            future_to_plan = {}
            
            for plan_item in processing_plan:
                file_info = plan_item["file_info"]
                required_stages = plan_item["required_stages"]
                
                if not required_stages:
                    # Skip files with no work needed
                    result = FileResult(
                        file_info=file_info,
                        success=True,
                        stages_completed=[],
                        total_processing_time=0.0
                    )
                    results.append(result)
                    continue
                
                future = executor.submit(
                    self.process_single_file,
                    file_info,
                    required_stages,
                    None  # No individual progress tracking in parallel mode
                )
                future_to_plan[future] = plan_item
            
            # Collect results as they complete
            for future in as_completed(future_to_plan):
                try:
                    result = future.result()
                    results.append(result)
                    completed_files += 1
                    
                    # Update overall progress
                    if progress_callback:
                        progress = ProgressInfo(
                            current=completed_files,
                            total=len(processing_plan),
                            stage="processing",
                            file_name=result.file_info.audio_path.name
                        )
                        progress_callback(progress)
                        
                except Exception as e:
                    plan_item = future_to_plan[future]
                    file_info = plan_item["file_info"]
                    
                    self.logger.error(
                        f"Parallel processing failed for {file_info.audio_path.name}: {e}",
                        file_path=str(file_info.audio_path),
                        error=str(e)
                    )
                    
                    result = FileResult(
                        file_info=file_info,
                        success=False,
                        stages_completed=[],
                        total_processing_time=0.0,
                        error=e
                    )
                    results.append(result)
        
        # Sort results to match input order
        results.sort(key=lambda r: str(r.file_info.audio_path))
        
        return results
    
    def _process_stage_with_retry(
        self,
        file_info: FileInfo,
        stage: ProcessingStage
    ) -> StageResult:
        """Process a single stage with retry logic.
        
        Args:
            file_info: File information
            stage: Stage to process
            
        Returns:
            StageResult: Stage processing result
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                return self._process_stage(file_info, stage)
                
            except Exception as e:
                retry_count += 1
                
                if retry_count > max_retries or not is_retryable_error(e):
                    # Max retries reached or non-retryable error
                    self.logger.error(
                        f"Stage {stage.value} failed permanently: {e}",
                        file_path=str(file_info.audio_path),
                        stage=stage.value,
                        retry_count=retry_count,
                        error=str(e)
                    )
                    
                    return StageResult(
                        stage=stage,
                        success=False,
                        error=e
                    )
                
                # Wait before retry
                delay = get_retry_delay(e)
                self.logger.warning(
                    f"Stage {stage.value} failed, retrying in {delay}s (attempt {retry_count}/{max_retries}): {e}",
                    file_path=str(file_info.audio_path),
                    stage=stage.value,
                    retry_count=retry_count,
                    retry_delay=delay,
                    error=str(e)
                )
                
                time.sleep(delay)
        
        # This should never be reached, but just in case
        return StageResult(
            stage=stage,
            success=False,
            error=ProcessingError(f"Max retries exceeded for stage {stage.value}")
        )
    
    def _process_stage(self, file_info: FileInfo, stage: ProcessingStage) -> StageResult:
        """Process a single stage.
        
        Args:
            file_info: File information  
            stage: Stage to process
            
        Returns:
            StageResult: Stage processing result
        """
        start_time = time.time()
        
        self.logger.log_file_processing(
            str(file_info.audio_path),
            stage.value,
            "started"
        )
        
        try:
            if stage == ProcessingStage.TRANSCRIPTION:
                result = self._process_transcription(file_info)
            elif stage == ProcessingStage.REFINEMENT:
                result = self._process_refinement(file_info)
            elif stage == ProcessingStage.SUMMARIZATION:
                result = self._process_summarization(file_info)
            else:
                raise ProcessingError(f"Unknown processing stage: {stage}")
            
            processing_time = time.time() - start_time
            
            self.logger.log_file_processing(
                str(file_info.audio_path),
                stage.value,
                "completed",
                processing_time=processing_time,
                output_path=str(result.output_path) if result.output_path else None
            )
            
            result.processing_time = processing_time
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            self.logger.log_file_processing(
                str(file_info.audio_path),
                stage.value,
                "failed",
                processing_time=processing_time,
                error=str(e)
            )
            
            return StageResult(
                stage=stage,
                success=False,
                processing_time=processing_time,
                error=e
            )
    
    def _process_transcription(self, file_info: FileInfo) -> StageResult:
        """Process transcription stage."""
        with self.logger.performance_timer(
            "transcription",
            file_path=str(file_info.audio_path),
            model=self.config.transcription.model_name
        ):
            # Perform transcription
            result = self.transcription_engine.transcribe(file_info.audio_path)
            
            # Save transcription
            self.file_manager.save_text_file(result.text, file_info.transcript_path)
            
            return StageResult(
                stage=ProcessingStage.TRANSCRIPTION,
                success=True,
                output_path=file_info.transcript_path,
                metadata=result.metadata
            )
    
    def _process_refinement(self, file_info: FileInfo) -> StageResult:
        """Process refinement stage."""
        # Read transcription
        transcript_text = self.file_manager.read_text_file(file_info.transcript_path)
        
        with self.logger.performance_timer(
            "refinement",
            file_path=str(file_info.audio_path),
            provider=self.config.llm.provider
        ):
            # Perform refinement
            result = self.llm_engine.refine_text(transcript_text)
            
            # Save refined text
            self.file_manager.save_text_file(result.refined_text, file_info.refined_path)
            
            return StageResult(
                stage=ProcessingStage.REFINEMENT,
                success=True,
                output_path=file_info.refined_path,
                metadata=result.metadata
            )
    
    def _process_summarization(self, file_info: FileInfo) -> StageResult:
        """Process summarization stage."""
        # Read refined text
        refined_text = self.file_manager.read_text_file(file_info.refined_path)
        
        with self.logger.performance_timer(
            "summarization",
            file_path=str(file_info.audio_path),
            provider=self.config.llm.provider
        ):
            # Perform summarization
            result = self.llm_engine.summarize_text(
                refined_text,
                audio_path=str(file_info.audio_path),
                original_txt_path=str(file_info.transcript_path),
                refined_txt_path=str(file_info.refined_path)
            )
            
            # Save summary
            self.file_manager.save_text_file(result.summary_markdown, file_info.summary_path)
            
            return StageResult(
                stage=ProcessingStage.SUMMARIZATION,
                success=True,
                output_path=file_info.summary_path,
                metadata=result.metadata
            )