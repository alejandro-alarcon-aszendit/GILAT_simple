"""Parallel processing service.

Handles all parallel workloads with clear separation of concerns and monitoring.
"""

import time
import concurrent.futures
from typing import List, Dict, Any, Callable
from dataclasses import dataclass

from src.core.config import ParallelConfig


@dataclass
class ParallelWorkload:
    """Represents a single parallel workload with metadata."""
    id: str
    name: str
    function: Callable
    args: tuple
    kwargs: dict
    estimated_duration: float = 0.0


@dataclass
class ParallelResult:
    """Result from a parallel workload execution."""
    workload_id: str
    success: bool
    result: Any = None
    error: str = None
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0


class ParallelProcessingService:
    """Service for managing and executing parallel workloads."""
    
    @staticmethod
    def execute_workloads(
        workloads: List[ParallelWorkload],
        max_workers: int = None,
        timeout: int = None
    ) -> Dict[str, Any]:
        """Execute multiple workloads in parallel with comprehensive monitoring.
        
        Args:
            workloads: List of workloads to execute
            max_workers: Maximum number of concurrent workers
            timeout: Timeout in seconds for the entire operation
            
        Returns:
            Dictionary containing results and performance metrics
        """
        if not workloads:
            return {
                "results": [],
                "performance": {
                    "total_time": 0.0,
                    "workloads_count": 0,
                    "successful_workloads": 0,
                    "failed_workloads": 0
                }
            }
        
        if max_workers is None:
            max_workers = min(len(workloads), ParallelConfig.MAX_TOPIC_WORKERS)
        
        if timeout is None:
            timeout = ParallelConfig.PROCESSING_TIMEOUT
        
        print(f"🚀 Starting parallel execution of {len(workloads)} workloads with {max_workers} workers...")
        start_time = time.time()
        
        results = []
        
        def execute_single_workload(workload: ParallelWorkload) -> ParallelResult:
            """Execute a single workload with error handling and timing."""
            workload_start = time.time()
            print(f"  📝 Processing workload: '{workload.name}' (ID: {workload.id})")
            
            try:
                result = workload.function(*workload.args, **workload.kwargs)
                workload_end = time.time()
                duration = workload_end - workload_start
                
                print(f"  ✅ Completed workload: '{workload.name}' in {duration:.2f}s")
                
                return ParallelResult(
                    workload_id=workload.id,
                    success=True,
                    result=result,
                    start_time=workload_start,
                    end_time=workload_end,
                    duration=duration
                )
                
            except Exception as e:
                workload_end = time.time()
                duration = workload_end - workload_start
                error_msg = str(e)
                
                print(f"  ❌ Failed workload: '{workload.name}' in {duration:.2f}s - {error_msg}")
                
                return ParallelResult(
                    workload_id=workload.id,
                    success=False,
                    error=error_msg,
                    start_time=workload_start,
                    end_time=workload_end,
                    duration=duration
                )
        
        # Execute workloads in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all workloads
            future_to_workload = {
                executor.submit(execute_single_workload, workload): workload 
                for workload in workloads
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_workload, timeout=timeout):
                try:
                    result = future.result()
                    results.append(result)
                except concurrent.futures.TimeoutError:
                    workload = future_to_workload[future]
                    print(f"  ⏱️  Timeout for workload: '{workload.name}'")
                    results.append(ParallelResult(
                        workload_id=workload.id,
                        success=False,
                        error="Timeout",
                        duration=timeout
                    ))
                except Exception as e:
                    workload = future_to_workload[future]
                    print(f"  ❌ Exception for workload: '{workload.name}' - {str(e)}")
                    results.append(ParallelResult(
                        workload_id=workload.id,
                        success=False,
                        error=f"Execution exception: {str(e)}",
                        duration=0.0
                    ))
        
        total_time = time.time() - start_time
        successful_count = sum(1 for r in results if r.success)
        failed_count = len(results) - successful_count
        
        print(f"🎉 Parallel execution completed in {total_time:.2f}s")
        print(f"    ✅ Successful: {successful_count}")
        print(f"    ❌ Failed: {failed_count}")
        
        # Calculate performance metrics
        individual_times = [r.duration for r in results if r.duration > 0]
        max_individual_time = max(individual_times) if individual_times else 0
        total_sequential_time = sum(individual_times) if individual_times else 0
        parallel_speedup = total_sequential_time / total_time if total_time > 0 else 1
        
        return {
            "results": results,
            "performance": {
                "total_time": total_time,
                "workloads_count": len(workloads),
                "successful_workloads": successful_count,
                "failed_workloads": failed_count,
                "max_workers": max_workers,
                "longest_individual_task": max_individual_time,
                "estimated_sequential_time": total_sequential_time,
                "speedup_factor": round(parallel_speedup, 2),
                "efficiency": round((total_sequential_time / (total_time * max_workers)) * 100, 1) if total_time > 0 else 0,
                "parallel_method": "ThreadPoolExecutor"
            }
        }
    
    @staticmethod
    def create_topic_workloads(
        topics: List[str],
        process_function: Callable,
        shared_kwargs: Dict[str, Any] = None
    ) -> List[ParallelWorkload]:
        """Create workloads for topic processing.
        
        Args:
            topics: List of topics to process
            process_function: Function to process each topic
            shared_kwargs: Common keyword arguments for all workloads
            
        Returns:
            List of ParallelWorkload objects
        """
        if shared_kwargs is None:
            shared_kwargs = {}
        
        workloads = []
        for i, topic in enumerate(topics):
            workloads.append(ParallelWorkload(
                id=f"topic_{i}",
                name=f"Topic: {topic}",
                function=process_function,
                args=(topic,),
                kwargs=shared_kwargs,
                estimated_duration=30.0  # Rough estimate for topic processing
            ))
        
        return workloads
    
    @staticmethod
    def monitor_workload_progress(results: List[ParallelResult]) -> Dict[str, Any]:
        """Monitor and analyze workload execution results.
        
        Args:
            results: List of ParallelResult objects
            
        Returns:
            Dictionary with monitoring metrics
        """
        if not results:
            return {"status": "no_results"}
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        durations = [r.duration for r in results if r.duration > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total_workloads": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) * 100 if results else 0,
            "average_duration": avg_duration,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "error_summary": [r.error for r in failed if r.error]
        } 