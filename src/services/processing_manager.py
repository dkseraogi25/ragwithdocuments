from typing import List, Optional, Dict, Any
import ray
import torch
import psutil
import threading
import os
from queue import Queue
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel

@dataclass
class ProcessingResources:
    cpu_count: int
    memory_available: float
    gpu_available: bool
    gpu_memory: Optional[float] = None

class ResourceManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._init_gpu()
        self._init_distributed()
        
    def _init_gpu(self):
        """Initialize GPU if available"""
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.device = torch.device("cuda")
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory
            self.logger.info(f"GPU initialized with {self.gpu_memory/1e9:.2f}GB memory")
        else:
            self.device = torch.device("cpu")
            self.logger.info("No GPU available, using CPU")
            
    def _init_distributed(self):
        """Initialize distributed processing with Ray"""
        import os
        import sys
        from pathlib import Path
        
        if not ray.is_initialized():
            # Add project root to PYTHONPATH for Ray workers
            project_root = str(Path(__file__).parent.parent.absolute())
            pythonpath = os.pathsep.join([project_root, os.environ.get('PYTHONPATH', '')])
            
            # Calculate resource limits
            total_memory = psutil.virtual_memory().total / (1024**3)  # GB
            per_worker_memory = total_memory * 0.02  # 2% of total system memory
            max_workers = 1  # Strictly limit to 1 worker
            
            runtime_env = {
                "env_vars": {
                    "PYTHONPATH": pythonpath,
                    "RAY_DISABLE_MEMORY_MONITOR": "0",  # Enable memory monitoring
                    "RAY_memory_monitor_refresh_ms": "1000"  # Check memory every second
                }
            }
            
            ray.init(
                ignore_reinit_error=True,
                runtime_env=runtime_env,
                namespace="rag_app",
                num_cpus=max_workers,
                _memory=per_worker_memory * 1024 * 1024 * 1024,  # Convert GB to bytes
                object_store_memory=int(per_worker_memory * 0.5 * 1024 * 1024 * 1024),  # 50% of per_worker_memory
                _system_config={
                    "object_spilling_threshold": 0.8,  # Spill objects when 80% full
                    "max_direct_call_object_size": 100 * 1024 * 1024,  # 100MB
                }
            )
            
            # Also add to current process's Python path
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
                
            self.logger.info(f"Distributed processing initialized with {max_workers} workers, {per_worker_memory:.1f}GB per worker")
            
    def get_available_resources(self) -> ProcessingResources:
        """Get current available computing resources"""
        resources = ProcessingResources(
            cpu_count=psutil.cpu_count(),
            memory_available=psutil.virtual_memory().available / (1024**3),  # GB
            gpu_available=self.gpu_available,
            gpu_memory=self.gpu_memory if self.gpu_available else None
        )
        return resources
        
    def get_optimal_batch_size(self, doc_size: int) -> int:
        """Determine optimal batch size based on available resources"""
        resources = self.get_available_resources()
        
        # Base batch size on available memory (both CPU and GPU if available)
        if self.gpu_available:
            # Use GPU memory for batch size calculation
            total_memory = resources.gpu_memory / (1024**3)  # Convert to GB
            batch_size = max(1, int((total_memory * 0.7) / (doc_size / 1024)))  # 70% of GPU memory
        else:
            # Use system memory for batch size calculation
            batch_size = max(1, int((resources.memory_available * 0.5) / (doc_size / 1024)))  # 50% of RAM
            
        return min(batch_size, 50)  # Cap at 50 to prevent overload

@ray.remote
class DistributedProcessor:
    """Distributed document processing using Ray"""
    
    def __init__(self):
        self.resource_manager = ResourceManager()
        self.model = None
        self.tokenizer = None
        
    def init_model(self, model_name: str):
        """Initialize the model on the worker"""
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            if self.resource_manager.gpu_available:
                self.model.to(self.resource_manager.device)
                
    def process_batch(self, texts: List[str], model_name: str = "sentence-transformers/all-mpnet-base-v2") -> List[Any]:
        """Process a batch of texts"""
        self.init_model(model_name)
        
        # Tokenize and process
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        if self.resource_manager.gpu_available:
            inputs = {k: v.to(self.resource_manager.device) for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()

class ProcessingOrchestrator:
    """Orchestrate distributed processing with auto-scaling"""
    
    def __init__(self, num_workers: Optional[int] = None):
        self.resource_manager = ResourceManager()
        # Strictly enforce single worker policy
        self.num_workers = 1  # Force single worker regardless of input
        self.processors = []
        self.active_workers = 0
        self.logger = logging.getLogger(__name__)
        self._initialize_workers()
        
    def _initialize_workers(self):
        """Initialize worker pool with memory monitoring"""
        self.processors = [DistributedProcessor.remote() for _ in range(self.num_workers)]
        self.active_workers = self.num_workers
        self.logger.info(f"Initialized {self.num_workers} workers")
        
    def _monitor_resources(self):
        """Monitor system resources"""
        mem_percent = psutil.virtual_memory().percent
        if mem_percent > 85:  # High memory usage
            self.logger.warning(f"High memory usage ({mem_percent}%), consider reducing batch size")
        self.active_workers = 1  # Always maintain single worker
        
    def process_documents(self, texts: List[str], batch_size: Optional[int] = None) -> List[Any]:
        """Process documents in parallel with auto-scaling"""
        if not texts:
            return []
            
        # Monitor and adjust resources
        self._monitor_resources()
            
        # Calculate optimal batch size if not provided
        if batch_size is None:
            avg_text_size = sum(len(text.encode('utf-8')) for text in texts) / len(texts)
            batch_size = self.resource_manager.get_optimal_batch_size(avg_text_size)
            # Increase batch size if we have fewer workers to maintain throughput
            batch_size = int(batch_size * (self.num_workers / self.active_workers))
            
        self.logger.info(f"Processing with batch size: {batch_size}, active workers: {self.active_workers}")
        
        # Split into batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        # Process batches in parallel with active workers only
        futures = []
        for i, batch in enumerate(batches):
            worker_idx = i % self.active_workers
            futures.append(self.processors[worker_idx].process_batch.remote(batch))
            
        # Get results with timeout to prevent hanging
        try:
            results = ray.get(futures, timeout=300)  # 5-minute timeout
        except Exception as e:
            self.logger.error(f"Error processing documents: {str(e)}")
            # Cleanup any hanging processes
            self._initialize_workers()
            raise
        
        # Flatten results
        return [item for batch in results for item in batch]
        
    def cleanup(self):
        """Clean up distributed resources"""
        try:
            # Force garbage collection first
            import gc
            gc.collect()
            
            # Kill all worker processes explicitly
            for processor in self.processors:
                ray.kill(processor)
            self.processors = []
            
            # Shutdown Ray
            if ray.is_initialized():
                ray.shutdown()
                
            self.logger.info("Distributed processing resources cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            
    def shutdown(self):
        """Shutdown Ray and cleanup resources"""
        try:
            self.cleanup()
            
            # Force terminate any remaining Python processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'ray::' in str(proc.cmdline()):
                        proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
            self.logger.info("Distributed processing shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
