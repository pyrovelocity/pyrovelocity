"""
Comprehensive component tensor flow testing framework.

This module provides systematic testing of tensor shapes and memory usage
throughout the component pipeline to identify remaining tensor reshaping issues.
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


class TensorFlowTracker:
    """Track tensor shapes and memory usage throughout component pipeline."""
    
    def __init__(self):
        self.snapshots = []
        self.memory_snapshots = []
        
    def track_context(self, stage: str, context: Dict[str, Any], memory_mb: float = None):
        """Track tensor shapes in context at a specific stage."""
        shapes = {}
        for key, value in context.items():
            if isinstance(value, torch.Tensor):
                shapes[key] = list(value.shape)
                
        self.snapshots.append({
            "stage": stage,
            "shapes": shapes,
            "memory_mb": memory_mb
        })
        
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
        
    def print_report(self):
        """Print comprehensive tensor flow report."""
        print("\n=== TENSOR FLOW REPORT ===")
        for snapshot in self.snapshots:
            print(f"\n--- {snapshot['stage']} ---")
            if snapshot['memory_mb']:
                print(f"Memory: {snapshot['memory_mb']:.2f} MB")
            for key, shape in snapshot['shapes'].items():
                print(f"  {key}: {shape}")
                
    def check_for_dimension_explosion(self, max_expected_dims: int = 3) -> List[str]:
        """Check for problematic tensor dimensions."""
        issues = []
        for snapshot in self.snapshots:
            for key, shape in snapshot['shapes'].items():
                if len(shape) > max_expected_dims:
                    issues.append(f"{snapshot['stage']}: {key} has {len(shape)} dimensions: {shape}")
                    
                # Check for very large dimensions
                for dim in shape:
                    if dim > 1000:
                        issues.append(f"{snapshot['stage']}: {key} has large dimension {dim}: {shape}")
                        
        return issues


@pytest.fixture
def test_data():
    """Create test data with controlled shapes."""
    batch_size = 30
    n_genes = 100
    
    # Create data with shapes that match full pipeline
    u_obs = torch.randint(0, 10, (batch_size, n_genes)).float()
    s_obs = torch.randint(0, 10, (batch_size, n_genes)).float()
    
    return {
        "u_obs": u_obs,
        "s_obs": s_obs,
    }


@pytest.fixture
def tracker():
    """Create a TensorFlowTracker instance."""
    return TensorFlowTracker()


def test_component_tensor_flow_step_by_step(test_data, tracker):
    """Test tensor flow through each component step by step."""
    # Initialize components
    prior_model = PiecewiseActivationPriorModel()
    dynamics_model = PiecewiseActivationDynamicsModel()
    likelihood_model = PiecewiseActivationPoissonLikelihoodModel()
    guide_model = AutoGuideFactory(guide_type="AutoNormal")
    
    # Clear Pyro state
    pyro.clear_param_store()
    
    # Start with input data
    context = dict(test_data)
    tracker.track_context("Input", context, tracker.get_memory_usage_mb())
    
    # Step 1: Prior
    context = prior_model.forward(context)
    tracker.track_context("After Prior", context, tracker.get_memory_usage_mb())
    
    # Step 2: Dynamics
    context = dynamics_model.forward(context)
    tracker.track_context("After Dynamics", context, tracker.get_memory_usage_mb())
    
    # Step 3: Likelihood
    context = likelihood_model.forward(context)
    tracker.track_context("After Likelihood", context, tracker.get_memory_usage_mb())
    
    # Print report
    tracker.print_report()
    
    # Check for dimension explosion
    issues = tracker.check_for_dimension_explosion(max_expected_dims=3)
    if issues:
        pytest.fail(f"Dimension explosion detected:\n" + "\n".join(issues))
    
    # Check expected tensor shapes
    assert context["u_obs"].shape == (30, 100)
    assert context["s_obs"].shape == (30, 100)
    assert context["u_expected"].shape == (30, 100)
    assert context["s_expected"].shape == (30, 100)


def test_full_model_tensor_flow(test_data, tracker):
    """Test tensor flow through full model execution."""
    # Create full model
    model = PyroVelocityModel(
        dynamics_model=PiecewiseActivationDynamicsModel(),
        prior_model=PiecewiseActivationPriorModel(),
        likelihood_model=PiecewiseActivationPoissonLikelihoodModel(),
        guide_model=AutoGuideFactory(guide_type="AutoNormal"),
    )
    
    # Clear Pyro state
    pyro.clear_param_store()
    
    # Track initial state
    tracker.track_context("Model Input", test_data, tracker.get_memory_usage_mb())
    
    # Execute full model
    result = model.forward(**test_data)
    tracker.track_context("Model Output", result, tracker.get_memory_usage_mb())
    
    # Print report
    tracker.print_report()
    
    # Check for dimension explosion
    issues = tracker.check_for_dimension_explosion(max_expected_dims=3)
    if issues:
        pytest.fail(f"Dimension explosion detected:\n" + "\n".join(issues))
    
    # Verify final shapes
    assert result["u_obs"].shape == (30, 100)
    assert result["s_obs"].shape == (30, 100)
    assert result["u_expected"].shape == (30, 100)
    assert result["s_expected"].shape == (30, 100)


def test_memory_usage_during_inference(test_data, tracker):
    """Test memory usage during inference with different batch sizes."""
    batch_sizes = [10, 30, 50, 100]
    memory_usage = []
    
    for batch_size in batch_sizes:
        # Create data with specific batch size
        u_obs = torch.randint(0, 10, (batch_size, 100)).float()
        s_obs = torch.randint(0, 10, (batch_size, 100)).float()
        data = {"u_obs": u_obs, "s_obs": s_obs}
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Create model
        model = PyroVelocityModel(
            dynamics_model=PiecewiseActivationDynamicsModel(),
            prior_model=PiecewiseActivationPriorModel(),
            likelihood_model=PiecewiseActivationPoissonLikelihoodModel(),
            guide_model=AutoGuideFactory(guide_type="AutoNormal"),
        )
        
        # Clear Pyro state
        pyro.clear_param_store()
        
        # Track memory before and after
        mem_before = tracker.get_memory_usage_mb()
        result = model.forward(**data)
        mem_after = tracker.get_memory_usage_mb()
        
        memory_usage.append({
            "batch_size": batch_size,
            "memory_before": mem_before,
            "memory_after": mem_after,
            "memory_delta": mem_after - mem_before
        })
        
        # Cleanup
        del model, result, data
        gc.collect()
    
    # Print memory usage report
    print("\n=== MEMORY USAGE REPORT ===")
    for usage in memory_usage:
        print(f"Batch size {usage['batch_size']}: "
              f"{usage['memory_before']:.2f} MB -> {usage['memory_after']:.2f} MB "
              f"(+{usage['memory_delta']:.2f} MB)")
    
    # Check for reasonable memory scaling
    # Memory should scale approximately linearly with batch size
    deltas = [usage['memory_delta'] for usage in memory_usage]
    
    # Check that memory doesn't explode (> 1GB per 100 samples is problematic)
    max_reasonable_delta = 1000  # 1GB
    for usage in memory_usage:
        if usage['memory_delta'] > max_reasonable_delta:
            pytest.fail(f"Memory usage too high for batch size {usage['batch_size']}: "
                       f"{usage['memory_delta']:.2f} MB")


def test_component_isolation_tensor_shapes(test_data, tracker):
    """Test each component in isolation to identify problematic components."""
    components = [
        ("Prior", PiecewiseActivationPriorModel()),
        ("Dynamics", PiecewiseActivationDynamicsModel()),
        ("Likelihood", PiecewiseActivationPoissonLikelihoodModel()),
    ]
    
    for name, component in components:
        # Clear Pyro state
        pyro.clear_param_store()
        
        # Start with fresh context
        context = dict(test_data)
        tracker.track_context(f"{name} Input", context)
        
        # Add required parameters for components that need them
        if name == "Dynamics":
            # Add parameters that would come from prior
            context.update({
                "alpha_off": torch.ones(100),
                "alpha_on": torch.ones(100) * 2.0,
                "gamma_star": torch.ones(100) * 0.5,
                "t_on_star": torch.ones(100) * 0.5,
                "delta_star": torch.ones(100) * 0.5,
                "t_star": torch.ones(100) * 1.0,
                "lambda_j": torch.ones(100) * 0.5,
            })
        elif name == "Likelihood":
            # Add parameters that would come from prior and dynamics
            context.update({
                "alpha_off": torch.ones(100),
                "alpha_on": torch.ones(100) * 2.0,
                "gamma_star": torch.ones(100) * 0.5,
                "t_on_star": torch.ones(100) * 0.5,
                "delta_star": torch.ones(100) * 0.5,
                "t_star": torch.ones(100) * 1.0,
                "lambda_j": torch.ones(100) * 0.5,
                "u_expected": torch.ones(30, 100),
                "s_expected": torch.ones(30, 100),
                "ut": torch.ones(30, 100),
                "st": torch.ones(30, 100),
            })
        
        # Execute component
        try:
            result = component.forward(context)
            tracker.track_context(f"{name} Output", result)
        except Exception as e:
            pytest.fail(f"Component {name} failed: {e}")
        
        # Check for dimension explosion in this component
        issues = tracker.check_for_dimension_explosion(max_expected_dims=3)
        component_issues = [issue for issue in issues if name in issue]
        if component_issues:
            pytest.fail(f"Dimension explosion in {name} component:\n" + 
                       "\n".join(component_issues))
    
    # Print final report
    tracker.print_report()


def test_guide_tensor_flow(test_data, tracker):
    """Test guide-specific tensor flow patterns."""
    # Create model and guide
    model = PyroVelocityModel(
        dynamics_model=PiecewiseActivationDynamicsModel(),
        prior_model=PiecewiseActivationPriorModel(),
        likelihood_model=PiecewiseActivationPoissonLikelihoodModel(),
        guide_model=AutoGuideFactory(guide_type="AutoNormal"),
    )
    
    # Clear Pyro state
    pyro.clear_param_store()
    
    # Track before model execution
    tracker.track_context("Before Model", test_data, tracker.get_memory_usage_mb())
    
    # Execute model to initialize guide
    _ = model.forward(**test_data)
    
    # Track after model execution
    tracker.track_context("After Model", test_data, tracker.get_memory_usage_mb())
    
    # Get guide
    guide = model.guide_model.get_guide()
    
    # Execute guide separately
    guide_result = guide(**test_data)
    
    # Track after guide execution
    tracker.track_context("After Guide", test_data, tracker.get_memory_usage_mb())
    
    # Print report
    tracker.print_report()
    
    # Check memory usage didn't explode
    memory_snapshots = [s['memory_mb'] for s in tracker.snapshots if s['memory_mb']]
    if memory_snapshots:
        max_memory = max(memory_snapshots)
        if max_memory > 2000:  # 2GB threshold
            pytest.fail(f"Memory usage too high: {max_memory:.2f} MB")