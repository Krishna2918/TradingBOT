"""
Async Job Scheduler
Implements asynchronous job scheduling for non-blocking operations
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from dataclasses import dataclass
from enum import Enum
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    """Job status types"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class JobPriority(Enum):
    """Job priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class JobType(Enum):
    """Job types"""
    DATA_FETCH = "data_fetch"
    MODEL_TRAINING = "model_training"
    BACKTESTING = "backtesting"
    RISK_CALCULATION = "risk_calculation"
    REPORT_GENERATION = "report_generation"
    CLEANUP = "cleanup"
    NOTIFICATION = "notification"
    CUSTOM = "custom"

@dataclass
class Job:
    """Job definition"""
    job_id: str
    job_type: JobType
    function: Callable
    args: Tuple
    kwargs: Dict[str, Any]
    priority: JobPriority
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    timeout: int = 300  # seconds
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: JobStatus = JobStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = None

@dataclass
class JobResult:
    """Job execution result"""
    job_id: str
    status: JobStatus
    result: Any
    error: Optional[str]
    execution_time: float
    timestamp: datetime

class AsyncJobScheduler:
    """Asynchronous job scheduler for non-blocking operations"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.job_queue = queue.PriorityQueue()
        self.jobs = {}  # job_id -> Job
        self.job_results = {}
        self.running_jobs = {}
        self.completed_jobs = []
        self.failed_jobs = []
        
        # Threading configuration
        self.max_workers = config.get('max_workers', 4)
        self.max_concurrent_jobs = config.get('max_concurrent_jobs', 8)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=min(2, self.max_workers))
        
        # Scheduler state
        self.scheduler_running = False
        self.scheduler_thread = None
        self.job_counter = 0
        
        # Job type configurations
        self.job_type_configs = {
            JobType.DATA_FETCH: {'max_retries': 5, 'timeout': 120, 'priority': JobPriority.HIGH},
            JobType.MODEL_TRAINING: {'max_retries': 2, 'timeout': 3600, 'priority': JobPriority.NORMAL},
            JobType.BACKTESTING: {'max_retries': 2, 'timeout': 1800, 'priority': JobPriority.NORMAL},
            JobType.RISK_CALCULATION: {'max_retries': 3, 'timeout': 300, 'priority': JobPriority.HIGH},
            JobType.REPORT_GENERATION: {'max_retries': 2, 'timeout': 600, 'priority': JobPriority.LOW},
            JobType.CLEANUP: {'max_retries': 1, 'timeout': 300, 'priority': JobPriority.LOW},
            JobType.NOTIFICATION: {'max_retries': 3, 'timeout': 30, 'priority': JobPriority.HIGH},
            JobType.CUSTOM: {'max_retries': 3, 'timeout': 300, 'priority': JobPriority.NORMAL}
        }
        
        logger.info("Async Job Scheduler initialized")
    
    def start_scheduler(self):
        """Start the job scheduler"""
        try:
            if self.scheduler_running:
                logger.warning("Scheduler is already running")
                return
            
            self.scheduler_running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()
            
            logger.info("Job scheduler started")
            
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
    
    def stop_scheduler(self):
        """Stop the job scheduler"""
        try:
            self.scheduler_running = False
            
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5)
            
            # Shutdown thread pools
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
            
            logger.info("Job scheduler stopped")
            
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
    
    def submit_job(self, job_type: JobType, function: Callable, 
                   args: Tuple = (), kwargs: Dict[str, Any] = None,
                   priority: JobPriority = None, scheduled_at: datetime = None,
                   **job_config) -> str:
        """Submit a job for execution"""
        try:
            # Generate job ID
            self.job_counter += 1
            job_id = f"job_{self.job_counter}_{uuid.uuid4().hex[:8]}"
            
            # Get job type configuration
            type_config = self.job_type_configs.get(job_type, {})
            
            # Set priority
            if priority is None:
                priority = type_config.get('priority', JobPriority.NORMAL)
            
            # Create job
            job = Job(
                job_id=job_id,
                job_type=job_type,
                function=function,
                args=args,
                kwargs=kwargs or {},
                priority=priority,
                max_retries=job_config.get('max_retries', type_config.get('max_retries', 3)),
                retry_delay=job_config.get('retry_delay', type_config.get('retry_delay', 60)),
                timeout=job_config.get('timeout', type_config.get('timeout', 300)),
                created_at=datetime.now(),
                scheduled_at=scheduled_at,
                metadata=job_config.get('metadata', {})
            )
            
            # Store job
            self.jobs[job_id] = job
            
            # Add to queue if not scheduled for later
            if scheduled_at is None or scheduled_at <= datetime.now():
                self._add_job_to_queue(job)
            
            logger.info(f"Job submitted: {job_id} ({job_type.value})")
            return job_id
            
        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            return None
    
    def _add_job_to_queue(self, job: Job):
        """Add job to priority queue"""
        try:
            # Priority queue uses negative priority for higher priority jobs
            priority_value = {
                JobPriority.CRITICAL: -4,
                JobPriority.HIGH: -3,
                JobPriority.NORMAL: -2,
                JobPriority.LOW: -1
            }[job.priority]
            
            self.job_queue.put((priority_value, job.created_at, job))
            
        except Exception as e:
            logger.error(f"Error adding job to queue: {e}")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        try:
            while self.scheduler_running:
                try:
                    # Check for scheduled jobs
                    self._check_scheduled_jobs()
                    
                    # Process pending jobs if we have capacity
                    if len(self.running_jobs) < self.max_concurrent_jobs:
                        self._process_next_job()
                    
                    # Check running jobs for completion
                    self._check_running_jobs()
                    
                    # Sleep briefly to prevent busy waiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error in scheduler loop: {e}")
                    time.sleep(1)
            
        except Exception as e:
            logger.error(f"Fatal error in scheduler loop: {e}")
    
    def _check_scheduled_jobs(self):
        """Check for jobs that are ready to be scheduled"""
        try:
            current_time = datetime.now()
            ready_jobs = []
            
            for job in self.jobs.values():
                if (job.status == JobStatus.PENDING and 
                    job.scheduled_at and 
                    job.scheduled_at <= current_time):
                    ready_jobs.append(job)
            
            for job in ready_jobs:
                self._add_job_to_queue(job)
                job.status = JobStatus.PENDING
                
        except Exception as e:
            logger.error(f"Error checking scheduled jobs: {e}")
    
    def _process_next_job(self):
        """Process the next job in the queue"""
        try:
            if self.job_queue.empty():
                return
            
            # Get next job from queue
            _, _, job = self.job_queue.get_nowait()
            
            # Check if job is still valid
            if job.job_id not in self.jobs or job.status != JobStatus.PENDING:
                return
            
            # Start job execution
            self._execute_job(job)
            
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Error processing next job: {e}")
    
    def _execute_job(self, job: Job):
        """Execute a job"""
        try:
            # Update job status
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            self.running_jobs[job.job_id] = job
            
            # Choose execution method based on job type
            if job.job_type in [JobType.MODEL_TRAINING, JobType.BACKTESTING]:
                # Use process pool for CPU-intensive tasks
                future = self.process_pool.submit(self._run_job, job)
            else:
                # Use thread pool for I/O-bound tasks
                future = self.thread_pool.submit(self._run_job, job)
            
            # Store future for later checking
            job.metadata['future'] = future
            
            logger.info(f"Job started: {job.job_id} ({job.job_type.value})")
            
        except Exception as e:
            logger.error(f"Error executing job {job.job_id}: {e}")
            self._handle_job_failure(job, str(e))
    
    def _run_job(self, job: Job) -> JobResult:
        """Run a job and return result"""
        try:
            start_time = time.time()
            
            # Execute the job function
            result = job.function(*job.args, **job.kwargs)
            
            execution_time = time.time() - start_time
            
            # Create job result
            job_result = JobResult(
                job_id=job.job_id,
                status=JobStatus.COMPLETED,
                result=result,
                error=None,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
            return job_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Create job result for failure
            job_result = JobResult(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                result=None,
                error=str(e),
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
            return job_result
    
    def _check_running_jobs(self):
        """Check running jobs for completion"""
        try:
            completed_jobs = []
            
            for job_id, job in self.running_jobs.items():
                future = job.metadata.get('future')
                if future and future.done():
                    try:
                        job_result = future.result()
                        self._handle_job_completion(job, job_result)
                        completed_jobs.append(job_id)
                    except Exception as e:
                        self._handle_job_failure(job, str(e))
                        completed_jobs.append(job_id)
                elif job.started_at and (datetime.now() - job.started_at).total_seconds() > job.timeout:
                    # Job timeout
                    self._handle_job_failure(job, "Job timeout")
                    completed_jobs.append(job_id)
            
            # Remove completed jobs from running jobs
            for job_id in completed_jobs:
                del self.running_jobs[job_id]
                
        except Exception as e:
            logger.error(f"Error checking running jobs: {e}")
    
    def _handle_job_completion(self, job: Job, job_result: JobResult):
        """Handle job completion"""
        try:
            # Update job status
            job.status = job_result.status
            job.completed_at = job_result.timestamp
            job.result = job_result.result
            job.error = job_result.error
            
            # Store result
            self.job_results[job.job_id] = job_result
            
            if job_result.status == JobStatus.COMPLETED:
                self.completed_jobs.append(job)
                logger.info(f"Job completed: {job.job_id} (execution time: {job_result.execution_time:.2f}s)")
            else:
                self.failed_jobs.append(job)
                logger.error(f"Job failed: {job.job_id} - {job_result.error}")
                
                # Handle retry if applicable
                if job.retry_count < job.max_retries:
                    self._schedule_job_retry(job)
            
        except Exception as e:
            logger.error(f"Error handling job completion: {e}")
    
    def _handle_job_failure(self, job: Job, error: str):
        """Handle job failure"""
        try:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now()
            job.error = error
            
            # Create failure result
            job_result = JobResult(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                result=None,
                error=error,
                execution_time=0.0,
                timestamp=datetime.now()
            )
            
            self.job_results[job.job_id] = job_result
            self.failed_jobs.append(job)
            
            logger.error(f"Job failed: {job.job_id} - {error}")
            
            # Handle retry if applicable
            if job.retry_count < job.max_retries:
                self._schedule_job_retry(job)
            
        except Exception as e:
            logger.error(f"Error handling job failure: {e}")
    
    def _schedule_job_retry(self, job: Job):
        """Schedule job retry"""
        try:
            job.retry_count += 1
            job.status = JobStatus.RETRYING
            
            # Schedule retry
            retry_time = datetime.now() + timedelta(seconds=job.retry_delay)
            job.scheduled_at = retry_time
            
            logger.info(f"Job retry scheduled: {job.job_id} (attempt {job.retry_count}/{job.max_retries})")
            
        except Exception as e:
            logger.error(f"Error scheduling job retry: {e}")
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        try:
            if job_id not in self.jobs:
                return False
            
            job = self.jobs[job_id]
            
            if job.status == JobStatus.RUNNING:
                # Try to cancel the future
                future = job.metadata.get('future')
                if future:
                    future.cancel()
            
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            
            # Remove from running jobs if present
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
            
            logger.info(f"Job cancelled: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling job: {e}")
            return False
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get job status"""
        try:
            if job_id not in self.jobs:
                return None
            
            job = self.jobs[job_id]
            job_result = self.job_results.get(job_id)
            
            status = {
                'job_id': job_id,
                'job_type': job.job_type.value,
                'status': job.status.value,
                'priority': job.priority.value,
                'created_at': job.created_at,
                'started_at': job.started_at,
                'completed_at': job.completed_at,
                'retry_count': job.retry_count,
                'max_retries': job.max_retries
            }
            
            if job_result:
                status.update({
                    'execution_time': job_result.execution_time,
                    'error': job_result.error
                })
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return None
    
    def get_job_result(self, job_id: str) -> Optional[Any]:
        """Get job result"""
        try:
            job_result = self.job_results.get(job_id)
            return job_result.result if job_result else None
            
        except Exception as e:
            logger.error(f"Error getting job result: {e}")
            return None
    
    def get_scheduler_statistics(self) -> Dict:
        """Get scheduler statistics"""
        try:
            total_jobs = len(self.jobs)
            pending_jobs = len([j for j in self.jobs.values() if j.status == JobStatus.PENDING])
            running_jobs = len(self.running_jobs)
            completed_jobs = len(self.completed_jobs)
            failed_jobs = len(self.failed_jobs)
            
            # Calculate success rate
            total_finished = completed_jobs + failed_jobs
            success_rate = (completed_jobs / total_finished * 100) if total_finished > 0 else 0
            
            # Calculate average execution time
            execution_times = [r.execution_time for r in self.job_results.values() if r.execution_time > 0]
            avg_execution_time = np.mean(execution_times) if execution_times else 0
            
            # Job type breakdown
            job_type_counts = {}
            for job in self.jobs.values():
                job_type = job.job_type.value
                job_type_counts[job_type] = job_type_counts.get(job_type, 0) + 1
            
            return {
                'total_jobs': total_jobs,
                'pending_jobs': pending_jobs,
                'running_jobs': running_jobs,
                'completed_jobs': completed_jobs,
                'failed_jobs': failed_jobs,
                'success_rate': success_rate,
                'average_execution_time': avg_execution_time,
                'job_type_breakdown': job_type_counts,
                'queue_size': self.job_queue.qsize(),
                'scheduler_running': self.scheduler_running,
                'max_concurrent_jobs': self.max_concurrent_jobs,
                'max_workers': self.max_workers
            }
            
        except Exception as e:
            logger.error(f"Error getting scheduler statistics: {e}")
            return {}
    
    def wait_for_job(self, job_id: str, timeout: int = None) -> Optional[Any]:
        """Wait for a job to complete and return its result"""
        try:
            if job_id not in self.jobs:
                return None
            
            job = self.jobs[job_id]
            start_time = time.time()
            
            while job.status in [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.RETRYING]:
                if timeout and (time.time() - start_time) > timeout:
                    return None
                
                time.sleep(0.1)
            
            return self.get_job_result(job_id)
            
        except Exception as e:
            logger.error(f"Error waiting for job: {e}")
            return None
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old completed and failed jobs"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            # Clean up old jobs
            old_job_ids = []
            for job_id, job in self.jobs.items():
                if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED] and
                    job.completed_at and job.completed_at < cutoff_time):
                    old_job_ids.append(job_id)
            
            for job_id in old_job_ids:
                del self.jobs[job_id]
                if job_id in self.job_results:
                    del self.job_results[job_id]
            
            # Clean up completed and failed job lists
            self.completed_jobs = [j for j in self.completed_jobs if j.job_id not in old_job_ids]
            self.failed_jobs = [j for j in self.failed_jobs if j.job_id not in old_job_ids]
            
            logger.info(f"Cleaned up {len(old_job_ids)} old jobs")
            
        except Exception as e:
            logger.error(f"Error cleaning up old jobs: {e}")
    
    def export_job_data(self, filepath: str, time_period_hours: int = 24):
        """Export job data to file"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_period_hours)
            
            # Filter recent jobs
            recent_jobs = [
                job for job in self.jobs.values()
                if job.created_at >= cutoff_time
            ]
            
            # Prepare export data
            export_data = {
                'export_info': {
                    'time_period_hours': time_period_hours,
                    'cutoff_time': cutoff_time.isoformat(),
                    'export_timestamp': datetime.now().isoformat(),
                    'total_jobs': len(recent_jobs)
                },
                'jobs': [
                    {
                        'job_id': job.job_id,
                        'job_type': job.job_type.value,
                        'status': job.status.value,
                        'priority': job.priority.value,
                        'created_at': job.created_at.isoformat(),
                        'started_at': job.started_at.isoformat() if job.started_at else None,
                        'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                        'retry_count': job.retry_count,
                        'max_retries': job.max_retries,
                        'timeout': job.timeout,
                        'metadata': job.metadata
                    }
                    for job in recent_jobs
                ],
                'results': [
                    {
                        'job_id': result.job_id,
                        'status': result.status.value,
                        'execution_time': result.execution_time,
                        'timestamp': result.timestamp.isoformat(),
                        'error': result.error
                    }
                    for result in self.job_results.values()
                    if result.timestamp >= cutoff_time
                ],
                'statistics': self.get_scheduler_statistics()
            }
            
            # Export to file
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported job data to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting job data: {e}")
