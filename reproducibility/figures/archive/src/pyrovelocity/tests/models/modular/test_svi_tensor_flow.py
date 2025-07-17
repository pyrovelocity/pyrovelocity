"""
SVI-specific tensor flow testing.

This module tests tensor flow during SVI training to identify where memory
issues occur during the training process.
"""

import gc
import tracemalloc
from typing import Any, Dict, List

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
from pyrovelocity.models.modular.inference.svi import create_svi, svi_step


class SVITensorFlowTracker:
    """Track tensor shapes and memory during SVI training."""
    
    def __init__(self):
        self.snapshots = []
        self.memory_snapshots = []
        
    def track_svi_step(self, step: int, loss: float, memory_mb: float = None):
        """Track SVI step results."""
        # Get parameter shapes from Pyro param store
        param_shapes = {}
        param_store = pyro.get_param_store()
        for name, param in param_store.items():
            if isinstance(param, torch.Tensor):
                param_shapes[name] = list(param.shape)
                
        self.snapshots.append({
            "step": step,
            "loss": loss,
            "param_shapes": param_shapes,
            "memory_mb": memory_mb
        })
        
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
        
    def print_svi_report(self):
        """Print SVI tensor flow report."""
        print("\n=== SVI TENSOR FLOW REPORT ===")
        for snapshot in self.snapshots:
            print(f"\n--- Step {snapshot['step']} ---")
            print(f"Loss: {snapshot['loss']:.4f}")
            if snapshot['memory_mb']:
                print(f"Memory: {snapshot['memory_mb']:.2f} MB")
            print("Parameter shapes:")
            for name, shape in snapshot['param_shapes'].items():
                print(f"  {name}: {shape}")
                
    def check_for_parameter_explosion(self) -> List[str]:
        """Check for parameter dimension explosion during training."""
        issues = []
        for snapshot in self.snapshots:
            for name, shape in snapshot['param_shapes'].items():
                # Check for very high dimensional parameters
                if len(shape) > 3:
                    issues.append(f"Step {snapshot['step']}: {name} has {len(shape)} dimensions: {shape}")
                    
                # Check for very large tensor sizes
                total_size = 1
                for dim in shape:
                    total_size *= dim
                if total_size > 100000:  # 100k elements
                    issues.append(f"Step {snapshot['step']}: {name} has {total_size} elements: {shape}")
                    
        return issues


@pytest.fixture
def svi_tracker():
    """Create an SVITensorFlowTracker instance."""
    return SVITensorFlowTracker()


@pytest.fixture
def svi_test_data():
    """Create test data for SVI testing."""
    batch_size = 30
    n_genes = 100
    
    u_obs = torch.randint(0, 10, (batch_size, n_genes)).float()
    s_obs = torch.randint(0, 10, (batch_size, n_genes)).float()
    
    return {
        "u_obs": u_obs,
        "s_obs": s_obs,
    }


def test_svi_initialization_tensor_flow(svi_test_data, svi_tracker):
    """Test tensor flow during SVI initialization."""
    # Create model
    model = PyroVelocityModel(
        dynamics_model=PiecewiseActivationDynamicsModel(),
        prior_model=PiecewiseActivationPriorModel(),
        likelihood_model=PiecewiseActivationPoissonLikelihoodModel(),
        guide_model=AutoGuideFactory(guide_type="AutoNormal"),
    )
    
    # Clear Pyro state
    pyro.clear_param_store()
    
    # Track memory before SVI creation
    memory_before = svi_tracker.get_memory_usage_mb()
    
    # Create guide
    guide = model.guide_model.create_guide(model.forward)
    
    # Track memory after guide creation
    memory_after_guide = svi_tracker.get_memory_usage_mb()
    
    # Create SVI
    svi = create_svi(
        model=model.forward,
        guide=guide,
        optimizer="adam",
        learning_rate=0.01,
        max_epochs=10,
    )
    
    # Track memory after SVI creation
    memory_after_svi = svi_tracker.get_memory_usage_mb()
    
    # Take one SVI step to initialize parameters
    loss = svi_step(svi, **svi_test_data)
    
    # Track memory after first step
    memory_after_step = svi_tracker.get_memory_usage_mb()
    
    # Track the step
    svi_tracker.track_svi_step(0, float(loss), memory_after_step)
    
    # Print memory report
    print(f"\nMemory usage during SVI initialization:")
    print(f"Before SVI: {memory_before:.2f} MB")
    print(f"After guide: {memory_after_guide:.2f} MB (+{memory_after_guide - memory_before:.2f} MB)")
    print(f"After SVI: {memory_after_svi:.2f} MB (+{memory_after_svi - memory_after_guide:.2f} MB)")
    print(f"After step: {memory_after_step:.2f} MB (+{memory_after_step - memory_after_svi:.2f} MB)")
    
    # Print parameter report
    svi_tracker.print_svi_report()
    
    # Check for parameter explosion
    issues = svi_tracker.check_for_parameter_explosion()
    if issues:
        pytest.fail(f"Parameter explosion detected:\n" + "\n".join(issues))
    
    # Check memory doesn't explode (more than 100MB increase is suspicious)
    total_memory_increase = memory_after_step - memory_before
    if total_memory_increase > 100:
        pytest.fail(f"Memory increase too large: {total_memory_increase:.2f} MB")


def test_svi_training_tensor_flow(svi_test_data, svi_tracker):
    """Test tensor flow during multiple SVI training steps."""
    # Create model
    model = PyroVelocityModel(
        dynamics_model=PiecewiseActivationDynamicsModel(),
        prior_model=PiecewiseActivationPriorModel(),
        likelihood_model=PiecewiseActivationPoissonLikelihoodModel(),
        guide_model=AutoGuideFactory(guide_type="AutoNormal"),
    )
    
    # Clear Pyro state
    pyro.clear_param_store()
    
    # Create guide and SVI
    guide = model.guide_model.create_guide(model.forward)
    svi = create_svi(
        model=model.forward,
        guide=guide,
        optimizer="adam",
        learning_rate=0.01,
    )
    
    # Track memory and parameters for multiple steps
    initial_memory = svi_tracker.get_memory_usage_mb()
    
    # Run several training steps
    for step in range(5):
        loss = svi_step(svi, **svi_test_data)
        current_memory = svi_tracker.get_memory_usage_mb()
        svi_tracker.track_svi_step(step, float(loss), current_memory)
        
        # Force garbage collection between steps
        gc.collect()
    
    # Print report
    svi_tracker.print_svi_report()
    
    # Check for parameter explosion
    issues = svi_tracker.check_for_parameter_explosion()
    if issues:
        pytest.fail(f"Parameter explosion detected:\n" + "\n".join(issues))
    
    # Check memory growth isn't excessive
    final_memory = svi_tracker.snapshots[-1]["memory_mb"]
    memory_growth = final_memory - initial_memory
    if memory_growth > 50:  # 50MB growth over 5 steps is too much
        pytest.fail(f"Memory growth too large: {memory_growth:.2f} MB over 5 steps")


def test_svi_guide_parameter_analysis(svi_test_data, svi_tracker):
    """Analyze guide parameter creation in detail."""
    # Create model
    model = PyroVelocityModel(
        dynamics_model=PiecewiseActivationDynamicsModel(),
        prior_model=PiecewiseActivationPriorModel(),
        likelihood_model=PiecewiseActivationPoissonLikelihoodModel(),
        guide_model=AutoGuideFactory(guide_type="AutoLowRankMultivariateNormal"),
    )
    
    # Clear Pyro state
    pyro.clear_param_store()
    
    # Track memory before guide creation
    memory_before = svi_tracker.get_memory_usage_mb()
    print(f"Memory before guide creation: {memory_before:.2f} MB")
    
    # Create guide
    guide = model.guide_model.create_guide(model.forward)
    memory_after_guide = svi_tracker.get_memory_usage_mb()
    print(f"Memory after guide creation: {memory_after_guide:.2f} MB")
    
    # Execute model once to see what parameters are created
    with pyro.poutine.trace() as tr:
        model.forward(**svi_test_data)
    
    memory_after_model = svi_tracker.get_memory_usage_mb()
    print(f"Memory after model execution: {memory_after_model:.2f} MB")
    
    # Execute guide once to see what parameters are created
    with pyro.poutine.trace() as guide_tr:
        guide(**svi_test_data)
    
    memory_after_guide_exec = svi_tracker.get_memory_usage_mb()
    print(f"Memory after guide execution: {memory_after_guide_exec:.2f} MB")
    
    # Print trace information
    print("\n=== MODEL TRACE ===")
    for name, node in tr.trace.nodes.items():
        if hasattr(node, 'value') and isinstance(node['value'], torch.Tensor):
            print(f"{name}: {list(node['value'].shape)}")
    
    print("\n=== GUIDE TRACE ===")
    for name, node in guide_tr.trace.nodes.items():
        if hasattr(node, 'value') and isinstance(node['value'], torch.Tensor):
            print(f"{name}: {list(node['value'].shape)}")
    
    # Print parameter store
    print("\n=== PARAMETER STORE ===")
    param_store = pyro.get_param_store()
    total_params = 0
    for name, param in param_store.items():
        if isinstance(param, torch.Tensor):
            param_size = param.numel()
            total_params += param_size
            print(f"{name}: {list(param.shape)} ({param_size} elements)")
    
    print(f"\nTotal parameters: {total_params}")
    
    # Check for suspiciously large parameters
    for name, param in param_store.items():
        if isinstance(param, torch.Tensor):
            if param.numel() > 50000:  # More than 50k elements is suspicious
                pytest.fail(f"Parameter {name} too large: {param.numel()} elements, shape {list(param.shape)}")


def test_svi_posterior_sampling_tensor_flow(svi_test_data, svi_tracker):
    """Test tensor flow during posterior sampling."""
    # Create model
    model = PyroVelocityModel(
        dynamics_model=PiecewiseActivationDynamicsModel(),
        prior_model=PiecewiseActivationPriorModel(),
        likelihood_model=PiecewiseActivationPoissonLikelihoodModel(),
        guide_model=AutoGuideFactory(guide_type="AutoNormal"),
    )
    
    # Clear Pyro state
    pyro.clear_param_store()
    
    # Create guide and run one training step
    guide = model.guide_model.create_guide(model.forward)
    svi = create_svi(model=model.forward, guide=guide, optimizer="adam")
    
    # Train for a few steps
    for _ in range(3):
        svi_step(svi, **svi_test_data)
    
    # Track memory before sampling
    memory_before = svi_tracker.get_memory_usage_mb()
    
    # Try to sample from posterior
    try:
        from pyrovelocity.models.modular.inference.svi import extract_posterior_samples
        
        # Sample with small number of samples first
        samples = extract_posterior_samples(
            guide=guide,
            num_samples=10,  # Small number to start
            model_args=(),
            model_kwargs=svi_test_data
        )
        
        memory_after = svi_tracker.get_memory_usage_mb()
        
        print(f"\nPosterior sampling memory usage:")
        print(f"Before sampling: {memory_before:.2f} MB")
        print(f"After sampling: {memory_after:.2f} MB")
        print(f"Memory increase: {memory_after - memory_before:.2f} MB")
        
        print(f"\nSample shapes:")
        for name, value in samples.items():
            if isinstance(value, torch.Tensor):
                print(f"{name}: {list(value.shape)}")
        
        # Check for dimension explosion in samples
        for name, value in samples.items():
            if isinstance(value, torch.Tensor):
                if len(value.shape) > 4:  # More than 4 dimensions is suspicious
                    pytest.fail(f"Sample {name} has too many dimensions: {value.shape}")
                    
                if value.numel() > 100000:  # More than 100k elements is suspicious
                    pytest.fail(f"Sample {name} too large: {value.numel()} elements")
    
    except Exception as e:
        pytest.fail(f"Posterior sampling failed: {e}")


def test_svi_memory_leak_detection(svi_test_data, svi_tracker):
    """Test for memory leaks during repeated SVI steps."""
    # Create model
    model = PyroVelocityModel(
        dynamics_model=PiecewiseActivationDynamicsModel(),
        prior_model=PiecewiseActivationPriorModel(),
        likelihood_model=PiecewiseActivationPoissonLikelihoodModel(),
        guide_model=AutoGuideFactory(guide_type="AutoNormal"),
    )
    
    memory_snapshots = []
    
    # Run multiple training sessions
    for session in range(3):
        # Clear Pyro state
        pyro.clear_param_store()
        gc.collect()
        
        # Create guide and SVI
        guide = model.guide_model.create_guide(model.forward)
        svi = create_svi(model=model.forward, guide=guide, optimizer="adam")
        
        # Take memory snapshot before training
        memory_before = svi_tracker.get_memory_usage_mb()
        
        # Run training
        for step in range(5):
            svi_step(svi, **svi_test_data)
        
        # Take memory snapshot after training
        memory_after = svi_tracker.get_memory_usage_mb()
        
        memory_snapshots.append({
            "session": session,
            "before": memory_before,
            "after": memory_after,
            "delta": memory_after - memory_before
        })
        
        # Clean up
        del guide, svi
        
    # Print memory report
    print("\n=== MEMORY LEAK DETECTION ===")
    for snapshot in memory_snapshots:
        print(f"Session {snapshot['session']}: "
              f"{snapshot['before']:.2f} MB -> {snapshot['after']:.2f} MB "
              f"(+{snapshot['delta']:.2f} MB)")
    
    # Check for memory leaks (memory should not keep growing between sessions)
    base_memory = memory_snapshots[0]["after"]
    for snapshot in memory_snapshots[1:]:
        memory_growth = snapshot["after"] - base_memory
        if memory_growth > 20:  # More than 20MB growth between sessions is suspicious
            pytest.fail(f"Potential memory leak: Session {snapshot['session']} "
                       f"uses {memory_growth:.2f} MB more than session 0")