"""
Memory profiling tests for modular components.

This module provides detailed memory profiling to identify where memory usage
explodes in the modular PyroVelocity pipeline.
"""

import gc
import tracemalloc
from typing import Any, Dict, List, Tuple

import psutil
import pyro
import pytest
import torch

from pyrovelocity.models.modular.components.dynamics import (
    PiecewiseActivationDynamicsModel,
)
from pyrovelocity.models.modular.components.guides import (
    AutoGuideFactory,
)
from pyrovelocity.models.modular.components.likelihoods import (
    PiecewiseActivationPoissonLikelihoodModel,
)
from pyrovelocity.models.modular.components.priors import (
    PiecewiseActivationPriorModel,
)
from pyrovelocity.models.modular.model import PyroVelocityModel


class MemoryProfiler:
    """Detailed memory profiling for component execution."""
    
    def __init__(self):
        self.snapshots = []
        self.tracemalloc_snapshots = []
        
    def start_tracing(self):
        """Start detailed memory tracing."""
        tracemalloc.start()
        
    def stop_tracing(self):
        """Stop detailed memory tracing."""
        tracemalloc.stop()
        
    def take_snapshot(self, label: str):
        """Take a memory snapshot."""
        # Get system memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get detailed memory stats if tracing is active
        tracemalloc_snapshot = None
        if tracemalloc.is_tracing():
            tracemalloc_snapshot = tracemalloc.take_snapshot()
            
        self.snapshots.append({
            "label": label,
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "tracemalloc_snapshot": tracemalloc_snapshot
        })
        
    def get_memory_growth(self) -> List[Tuple[str, float]]:
        """Get memory growth between snapshots."""
        if len(self.snapshots) < 2:
            return []
            
        growth = []
        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i-1]
            curr = self.snapshots[i]
            delta = curr["rss_mb"] - prev["rss_mb"]
            growth.append((f"{prev['label']} -> {curr['label']}", delta))
            
        return growth
        
    def print_memory_report(self):
        """Print detailed memory report."""
        print("\n=== MEMORY PROFILING REPORT ===")
        
        # Print snapshots
        for snapshot in self.snapshots:
            print(f"{snapshot['label']}: RSS={snapshot['rss_mb']:.2f} MB, "
                  f"VMS={snapshot['vms_mb']:.2f} MB")
                  
        # Print growth
        growth = self.get_memory_growth()
        if growth:
            print("\n--- Memory Growth ---")
            for transition, delta in growth:
                print(f"{transition}: {delta:+.2f} MB")
                
        # Print top memory consumers if tracing
        if self.snapshots and self.snapshots[-1]["tracemalloc_snapshot"]:
            print("\n--- Top Memory Consumers ---")
            snapshot = self.snapshots[-1]["tracemalloc_snapshot"]
            top_stats = snapshot.statistics('lineno')[:10]
            for stat in top_stats:
                print(f"{stat}")
                
    def check_memory_explosion(self, max_growth_mb: float = 500) -> List[str]:
        """Check for memory explosions."""
        issues = []
        growth = self.get_memory_growth()
        
        for transition, delta in growth:
            if delta > max_growth_mb:
                issues.append(f"Memory explosion: {transition} (+{delta:.2f} MB)")
                
        return issues


@pytest.fixture
def profiler():
    """Create a MemoryProfiler instance."""
    return MemoryProfiler()


@pytest.fixture
def realistic_data():
    """Create realistic data sizes for memory profiling."""
    # Use sizes that match actual validation pipeline
    batch_size = 200
    n_genes = 100
    
    u_obs = torch.randint(0, 10, (batch_size, n_genes)).float()
    s_obs = torch.randint(0, 10, (batch_size, n_genes)).float()
    
    return {
        "u_obs": u_obs,
        "s_obs": s_obs,
    }


def test_memory_profile_component_pipeline(realistic_data, profiler):
    """Profile memory usage through the component pipeline."""
    # Start detailed tracing
    profiler.start_tracing()
    
    # Clean up before starting
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Take initial snapshot
    profiler.take_snapshot("Initial")
    
    # Initialize components
    prior_model = PiecewiseActivationPriorModel()
    profiler.take_snapshot("After Prior Init")
    
    dynamics_model = PiecewiseActivationDynamicsModel()
    profiler.take_snapshot("After Dynamics Init")
    
    likelihood_model = PiecewiseActivationPoissonLikelihoodModel()
    profiler.take_snapshot("After Likelihood Init")
    
    guide_model = AutoGuideFactory(guide_type="AutoNormal")
    profiler.take_snapshot("After Guide Init")
    
    # Clear Pyro state
    pyro.clear_param_store()
    profiler.take_snapshot("After Pyro Clear")
    
    # Execute components step by step
    context = dict(realistic_data)
    profiler.take_snapshot("Before Prior Forward")
    
    context = prior_model.forward(context)
    profiler.take_snapshot("After Prior Forward")
    
    context = dynamics_model.forward(context)
    profiler.take_snapshot("After Dynamics Forward")
    
    context = likelihood_model.forward(context)
    profiler.take_snapshot("After Likelihood Forward")
    
    # Stop tracing
    profiler.stop_tracing()
    
    # Print report
    profiler.print_memory_report()
    
    # Check for memory explosions
    issues = profiler.check_memory_explosion(max_growth_mb=200)
    if issues:
        pytest.fail(f"Memory explosion detected:\n" + "\n".join(issues))


def test_memory_profile_full_model_execution(realistic_data, profiler):
    """Profile memory usage during full model execution."""
    # Start tracing
    profiler.start_tracing()
    
    # Clean up
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Take initial snapshot
    profiler.take_snapshot("Initial")
    
    # Create model
    model = PyroVelocityModel(
        dynamics_model=PiecewiseActivationDynamicsModel(),
        prior_model=PiecewiseActivationPriorModel(),
        likelihood_model=PiecewiseActivationPoissonLikelihoodModel(),
        guide_model=AutoGuideFactory(guide_type="AutoNormal"),
    )
    profiler.take_snapshot("After Model Creation")
    
    # Clear Pyro state
    pyro.clear_param_store()
    profiler.take_snapshot("After Pyro Clear")
    
    # Execute model
    result = model.forward(**realistic_data)
    profiler.take_snapshot("After Model Forward")
    
    # Stop tracing
    profiler.stop_tracing()
    
    # Print report
    profiler.print_memory_report()
    
    # Check for memory explosions
    issues = profiler.check_memory_explosion(max_growth_mb=500)
    if issues:
        pytest.fail(f"Memory explosion detected:\n" + "\n".join(issues))


def test_memory_profile_inference_simulation(realistic_data, profiler):
    """Profile memory usage during inference simulation."""
    # Start tracing
    profiler.start_tracing()
    
    # Clean up
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Take initial snapshot
    profiler.take_snapshot("Initial")
    
    # Create model
    model = PyroVelocityModel(
        dynamics_model=PiecewiseActivationDynamicsModel(),
        prior_model=PiecewiseActivationPriorModel(),
        likelihood_model=PiecewiseActivationPoissonLikelihoodModel(),
        guide_model=AutoGuideFactory(guide_type="AutoNormal"),
    )
    profiler.take_snapshot("After Model Creation")
    
    # Clear Pyro state
    pyro.clear_param_store()
    profiler.take_snapshot("After Pyro Clear")
    
    # Execute model to initialize guide
    _ = model.forward(**realistic_data)
    profiler.take_snapshot("After Model Forward")
    
    # Get guide
    guide = model.guide_model.get_guide()
    profiler.take_snapshot("After Guide Creation")
    
    # Execute guide multiple times (simulating inference)
    for i in range(5):
        _ = guide(**realistic_data)
        profiler.take_snapshot(f"After Guide Forward {i+1}")
    
    # Stop tracing
    profiler.stop_tracing()
    
    # Print report
    profiler.print_memory_report()
    
    # Check for memory explosions
    issues = profiler.check_memory_explosion(max_growth_mb=300)
    if issues:
        pytest.fail(f"Memory explosion detected:\n" + "\n".join(issues))


def test_memory_profile_batch_size_scaling(profiler):
    """Profile memory usage scaling with batch size."""
    batch_sizes = [50, 100, 200, 400]
    memory_usage = []
    
    for batch_size in batch_sizes:
        # Start tracing
        profiler.start_tracing()
        
        # Clean up
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Create data
        u_obs = torch.randint(0, 10, (batch_size, 100)).float()
        s_obs = torch.randint(0, 10, (batch_size, 100)).float()
        data = {"u_obs": u_obs, "s_obs": s_obs}
        
        # Take initial snapshot
        profiler.take_snapshot(f"Initial (batch={batch_size})")
        
        # Create model
        model = PyroVelocityModel(
            dynamics_model=PiecewiseActivationDynamicsModel(),
            prior_model=PiecewiseActivationPriorModel(),
            likelihood_model=PiecewiseActivationPoissonLikelihoodModel(),
            guide_model=AutoGuideFactory(guide_type="AutoNormal"),
        )
        
        # Clear Pyro state
        pyro.clear_param_store()
        
        # Execute model
        result = model.forward(**data)
        profiler.take_snapshot(f"After Forward (batch={batch_size})")
        
        # Stop tracing
        profiler.stop_tracing()
        
        # Get memory usage
        initial_memory = profiler.snapshots[-2]["rss_mb"]
        final_memory = profiler.snapshots[-1]["rss_mb"]
        memory_delta = final_memory - initial_memory
        
        memory_usage.append({
            "batch_size": batch_size,
            "initial_memory": initial_memory,
            "final_memory": final_memory,
            "memory_delta": memory_delta
        })
        
        # Clean up
        del model, result, data
        gc.collect()
        
        # Reset profiler for next iteration
        profiler.snapshots = []
    
    # Print scaling report
    print("\n=== BATCH SIZE SCALING REPORT ===")
    for usage in memory_usage:
        print(f"Batch size {usage['batch_size']}: "
              f"{usage['initial_memory']:.2f} MB -> {usage['final_memory']:.2f} MB "
              f"(+{usage['memory_delta']:.2f} MB)")
    
    # Check for reasonable scaling
    # Memory should scale roughly linearly with batch size
    per_sample_memory = []
    for usage in memory_usage:
        per_sample_mb = usage['memory_delta'] / usage['batch_size']
        per_sample_memory.append(per_sample_mb)
    
    print(f"\nMemory per sample: {per_sample_memory}")
    
    # Check for memory explosions (more than 10MB per sample is problematic)
    max_per_sample = max(per_sample_memory)
    if max_per_sample > 10:
        pytest.fail(f"Memory per sample too high: {max_per_sample:.2f} MB/sample")


def test_memory_profile_posterior_sampling(realistic_data, profiler):
    """Profile memory usage during posterior sampling simulation."""
    # Start tracing
    profiler.start_tracing()
    
    # Clean up
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Take initial snapshot
    profiler.take_snapshot("Initial")
    
    # Create model
    model = PyroVelocityModel(
        dynamics_model=PiecewiseActivationDynamicsModel(),
        prior_model=PiecewiseActivationPriorModel(),
        likelihood_model=PiecewiseActivationPoissonLikelihoodModel(),
        guide_model=AutoGuideFactory(guide_type="AutoNormal"),
    )
    profiler.take_snapshot("After Model Creation")
    
    # Clear Pyro state
    pyro.clear_param_store()
    profiler.take_snapshot("After Pyro Clear")
    
    # Execute model to initialize guide
    _ = model.forward(**realistic_data)
    profiler.take_snapshot("After Model Forward")
    
    # Simulate posterior sampling (multiple forward passes)
    n_samples = 10
    for i in range(n_samples):
        # Clear Pyro's trace
        pyro.clear_param_store()
        
        # Execute model
        result = model.forward(**realistic_data)
        
        if i % 2 == 0:  # Take snapshots every 2 samples
            profiler.take_snapshot(f"After Sample {i+1}")
    
    # Stop tracing
    profiler.stop_tracing()
    
    # Print report
    profiler.print_memory_report()
    
    # Check for memory explosions during sampling
    issues = profiler.check_memory_explosion(max_growth_mb=200)
    if issues:
        pytest.fail(f"Memory explosion during sampling:\n" + "\n".join(issues))