"""
DocuBot Background Task Queue System
Asynchronous task processing with priority and retry logic
"""

import queue
import threading
import time
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import traceback


class TaskPriority(Enum):
    """Task priority levels"""
    HIGH = 0
    NORMAL = 1
    LOW = 2


class TaskStatus(Enum):
    """Task status states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Task:
    """Background task definition"""
    id: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: Optional[float] = None
    
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    
    callback: Optional[Callable] = None
    callback_args: tuple = field(default_factory=tuple)
    callback_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskQueue:
    """Background task processing queue"""
    
    def __init__(self, max_workers: int = 4, persist_dir: Optional[Path] = None):
        self.max_workers = max_workers
        self.persist_dir = persist_dir or Path.home() / ".docubot" / "task_queue"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.pending_queue = queue.PriorityQueue()
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        
        self.workers: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        
        self._load_persisted_tasks()
        self._start_workers()
    
    def _start_workers(self):
        """Start worker threads"""
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"TaskWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self):
        """Worker thread main loop"""
        while not self._stop_event.is_set():
            try:
                priority, task = self.pending_queue.get(timeout=1.0)
                
                with self._lock:
                    task.status = TaskStatus.RUNNING
                    task.started_at = datetime.now()
                    self.running_tasks[task.id] = task
                
                try:
                    result = self._execute_task(task)
                    
                    with self._lock:
                        task.status = TaskStatus.COMPLETED
                        task.completed_at = datetime.now()
                        task.result = result
                        
                        self.completed_tasks[task.id] = task
                        if task.id in self.running_tasks:
                            del self.running_tasks[task.id]
                    
                    if task.callback:
                        try:
                            task.callback(
                                task.result,
                                *task.callback_args,
                                **task.callback_kwargs
                            )
                        except Exception as e:
                            print(f"Task callback error: {e}")
                    
                    self._persist_task(task)
                
                except Exception as e:
                    self._handle_task_failure(task, e)
                
                finally:
                    self.pending_queue.task_done()
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
    
    def _execute_task(self, task: Task) -> Any:
        """Execute a task with timeout handling"""
        if task.timeout:
            def task_wrapper():
                return task.function(*task.args, **task.kwargs)
            
            future = threading.Thread(target=task_wrapper)
            future.start()
            future.join(timeout=task.timeout)
            
            if future.is_alive():
                raise TimeoutError(f"Task timed out after {task.timeout} seconds")
            
            return task.result
        else:
            return task.function(*task.args, **task.kwargs)
    
    def _handle_task_failure(self, task: Task, error: Exception):
        """Handle task execution failure"""
        with self._lock:
            task.retry_count += 1
            task.error = str(error)
            
            if task.retry_count <= task.max_retries:
                task.status = TaskStatus.RETRYING
                print(f"Task {task.id} failed, retrying ({task.retry_count}/{task.max_retries}): {error}")
                
                time.sleep(task.retry_delay)
                self.add_task(task)
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                
                self.failed_tasks[task.id] = task
                if task.id in self.running_tasks:
                    del self.running_tasks[task.id]
                
                print(f"Task {task.id} failed after {task.max_retries} retries: {error}")
                
                error_details = {
                    'error': str(error),
                    'traceback': traceback.format_exc(),
                    'retry_count': task.retry_count,
                    'task_id': task.id
                }
                task.metadata['error_details'] = error_details
            
            self._persist_task(task)
    
    def add_task(
        self,
        function: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        task_id: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: Optional[float] = None,
        callback: Optional[Callable] = None,
        callback_args: tuple = (),
        callback_kwargs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a task to the queue"""
        if kwargs is None:
            kwargs = {}
        if callback_kwargs is None:
            callback_kwargs = {}
        if metadata is None:
            metadata = {}
        
        task_id = task_id or f"task_{int(time.time())}_{hash(function)}"
        
        task = Task(
            id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
            callback=callback,
            callback_args=callback_args,
            callback_kwargs=callback_kwargs,
            metadata=metadata
        )
        
        priority_value = priority.value
        
        with self._lock:
            self.pending_queue.put((priority_value, task))
            self._persist_task(task)
        
        return task_id
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[Any]:
        """Wait for a specific task to complete"""
        start_time = time.time()
        
        while True:
            with self._lock:
                if task_id in self.completed_tasks:
                    return self.completed_tasks[task_id].result
                
                if task_id in self.failed_tasks:
                    raise RuntimeError(f"Task {task_id} failed: {self.failed_tasks[task_id].error}")
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")
            
            time.sleep(0.1)
    
    def wait_for_all(self, timeout: Optional[float] = None):
        """Wait for all pending tasks to complete"""
        self.pending_queue.join()
        
        start_time = time.time()
        while self.running_tasks:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Timeout waiting for all tasks")
            time.sleep(0.1)
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a task"""
        with self._lock:
            if task_id in self.running_tasks:
                return self.running_tasks[task_id].status
            elif task_id in self.completed_tasks:
                return self.completed_tasks[task_id].status
            elif task_id in self.failed_tasks:
                return self.failed_tasks[task_id].status
        
        return None
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self._lock:
            return {
                'pending': self.pending_queue.qsize(),
                'running': len(self.running_tasks),
                'completed': len(self.completed_tasks),
                'failed': len(self.failed_tasks),
                'workers': len([w for w in self.workers if w.is_alive()]),
                'timestamp': datetime.now().isoformat()
            }
    
    def _persist_task(self, task: Task):
        """Persist task to disk"""
        try:
            task_file = self.persist_dir / f"{task.id}.json"
            
            task_dict = {
                'id': task.id,
                'status': task.status.value,
                'created_at': task.created_at.isoformat(),
                'started_at': task.started_at.isoformat() if task.started_at else None,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                'priority': task.priority.value,
                'max_retries': task.max_retries,
                'retry_count': task.retry_count,
                'error': task.error,
                'metadata': task.metadata
            }
            
            with open(task_file, 'w', encoding='utf-8') as f:
                json.dump(task_dict, f, indent=2)
        
        except Exception as e:
            print(f"Error persisting task {task.id}: {e}")
    
    def _load_persisted_tasks(self):
        """Load persisted tasks from disk"""
        try:
            for task_file in self.persist_dir.glob("*.json"):
                try:
                    with open(task_file, 'r', encoding='utf-8') as f:
                        task_dict = json.load(f)
                    
                    created_at = datetime.fromisoformat(task_dict['created_at'])
                    
                    if datetime.now() - created_at > timedelta(days=7):
                        task_file.unlink()
                        continue
                
                except Exception as e:
                    print(f"Error loading task file {task_file}: {e}")
                    continue
        
        except Exception as e:
            print(f"Error loading persisted tasks: {e}")
    
    def cleanup_old_tasks(self, max_age_days: int = 7):
        """Clean up old task files"""
        cutoff = datetime.now() - timedelta(days=max_age_days)
        
        for task_file in self.persist_dir.glob("*.json"):
            try:
                with open(task_file, 'r', encoding='utf-8') as f:
                    task_dict = json.load(f)
                
                created_at = datetime.fromisoformat(task_dict['created_at'])
                
                if created_at < cutoff:
                    task_file.unlink()
            
            except Exception:
                try:
                    task_file.unlink()
                except:
                    pass
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """Shutdown the task queue"""
        self._stop_event.set()
        
        if wait:
            start_time = time.time()
            
            for worker in self.workers:
                if worker.is_alive():
                    worker.join(timeout=max(0, timeout - (time.time() - start_time)) if timeout else None)
            
            self.workers.clear()


# Global task queue instance
_task_queue = None

def get_task_queue() -> TaskQueue:
    """Get singleton task queue instance"""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
    return _task_queue


def background_task(
    priority: TaskPriority = TaskPriority.NORMAL,
    task_id: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    timeout: Optional[float] = None
):
    """Decorator for running functions as background tasks"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            queue = get_task_queue()
            return queue.add_task(
                function=func,
                args=args,
                kwargs=kwargs,
                priority=priority,
                task_id=task_id,
                max_retries=max_retries,
                retry_delay=retry_delay,
                timeout=timeout
            )
        return wrapper
    return decorator
