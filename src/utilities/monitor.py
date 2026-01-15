"""
DocuBot Performance Monitoring System
Monitor memory usage, execution times, and system resources
"""

import os
import time
import psutil
import threading
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import statistics


@dataclass
class PerformanceMetric:
    """Single performance metric"""
    timestamp: datetime
    metric_type: str
    value: float
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)


class PerformanceMonitor:
    """Monitor system and application performance"""
    
    def __init__(self, monitor_dir: Optional[Path] = None):
        self.monitor_dir = monitor_dir or Path.home() / ".docubot" / "monitoring"
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics: List[PerformanceMetric] = []
        self._max_metrics = 10000
        self._lock = threading.RLock()
        
        self.metric_file = self.monitor_dir / "metrics.json"
        self._load_metrics()
        
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
    
    def _load_metrics(self):
        """Load saved metrics from disk"""
        if self.metric_file.exists():
            try:
                with open(self.metric_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.metrics = [
                    PerformanceMetric(
                        timestamp=datetime.fromisoformat(m['timestamp']),
                        metric_type=m['metric_type'],
                        value=m['value'],
                        unit=m['unit'],
                        tags=m.get('tags', {})
                    )
                    for m in data
                ]
            except Exception as e:
                print(f"Warning: Could not load metrics: {e}")
                self.metrics = []
    
    def _save_metrics(self):
        """Save metrics to disk"""
        with self._lock:
            data = [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'metric_type': m.metric_type,
                    'value': m.value,
                    'unit': m.unit,
                    'tags': m.tags
                }
                for m in self.metrics[-1000:]  # Save only recent metrics
            ]
            
            try:
                with open(self.metric_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save metrics: {e}")
    
    def record_metric(self, metric_type: str, value: float, unit: str = "", tags: Optional[Dict[str, str]] = None):
        """Record a performance metric"""
        with self._lock:
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                metric_type=metric_type,
                value=value,
                unit=unit,
                tags=tags or {}
            )
            
            self.metrics.append(metric)
            
            if len(self.metrics) > self._max_metrics:
                self.metrics = self.metrics[-self._max_metrics:]
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'total_mb': psutil.virtual_memory().total / 1024 / 1024,
            'used_percent': psutil.virtual_memory().percent
        }
    
    def get_cpu_usage(self) -> Dict[str, float]:
        """Get CPU usage statistics"""
        process = psutil.Process(os.getpid())
        
        return {
            'process_percent': process.cpu_percent(interval=0.1),
            'system_percent': psutil.cpu_percent(interval=0.1),
            'system_load': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
        }
    
    def get_disk_usage(self) -> Dict[str, float]:
        """Get disk usage statistics"""
        data_dir = Path.home() / ".docubot"
        
        if data_dir.exists():
            usage = psutil.disk_usage(str(data_dir))
            return {
                'total_gb': usage.total / 1024 / 1024 / 1024,
                'used_gb': usage.used / 1024 / 1024 / 1024,
                'free_gb': usage.free / 1024 / 1024 / 1024,
                'percent': usage.percent
            }
        
        return {}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get all system metrics"""
        return {
            'memory': self.get_memory_usage(),
            'cpu': self.get_cpu_usage(),
            'disk': self.get_disk_usage(),
            'timestamp': datetime.now().isoformat()
        }
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start background monitoring thread"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        
        def monitor_loop():
            while not self._stop_monitoring.wait(interval_seconds):
                try:
                    system_metrics = self.get_system_metrics()
                    
                    self.record_metric(
                        'memory_usage',
                        system_metrics['memory']['rss_mb'],
                        'MB',
                        {'type': 'process'}
                    )
                    
                    self.record_metric(
                        'cpu_usage',
                        system_metrics['cpu']['process_percent'],
                        'percent',
                        {'type': 'process'}
                    )
                    
                    self.record_metric(
                        'disk_usage',
                        system_metrics['disk'].get('percent', 0),
                        'percent',
                        {'type': 'system'}
                    )
                    
                    self._save_metrics()
                    
                except Exception as e:
                    print(f"Monitoring error: {e}")
        
        self._monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        if self._monitoring_thread:
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5)
            self._monitoring_thread = None
    
    def get_metric_summary(self, metric_type: str, hours: int = 24) -> Dict[str, Any]:
        """Get summary statistics for a metric type"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_metrics = [
                m for m in self.metrics
                if m.metric_type == metric_type and m.timestamp > cutoff
            ]
            
            if not recent_metrics:
                return {}
            
            values = [m.value for m in recent_metrics]
            
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                'latest': values[-1],
                'time_range_hours': hours
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system': self.get_system_metrics(),
            'metrics_summary': {}
        }
        
        metric_types = set(m.metric_type for m in self.metrics)
        
        for metric_type in metric_types:
            if metric_type:
                summary = self.get_metric_summary(metric_type, hours=1)
                if summary:
                    report['metrics_summary'][metric_type] = summary
        
        return report
    
    def clear_metrics(self):
        """Clear all stored metrics"""
        with self._lock:
            self.metrics = []
            
            if self.metric_file.exists():
                self.metric_file.unlink()


class ExecutionTimer:
    """Context manager for timing code execution"""
    
    def __init__(self, name: str, monitor: Optional[PerformanceMonitor] = None, tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.monitor = monitor
        self.tags = tags or {}
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        
        if self.monitor:
            self.monitor.record_metric(
                f"execution_time_{self.name}",
                self.elapsed,
                "seconds",
                {**self.tags, 'operation': self.name}
            )
        
        return False


# Global monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get singleton performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def time_execution(name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator for timing function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with ExecutionTimer(name, get_performance_monitor(), tags):
                return func(*args, **kwargs)
        return wrapper
    return decorator
