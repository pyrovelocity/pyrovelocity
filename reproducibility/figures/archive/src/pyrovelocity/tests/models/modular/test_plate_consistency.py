"""Test plate consistency between model components."""
import pytest
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive
from pyrovelocity.models.modular.factory import create_piecewise_activation_model


def test_plate_name_consistency_in_predictive():
    """Test that consistent plate names work correctly in Predictive context."""
    pyro.clear_param_store()
    pyro.set_rng_seed(42)
    
    # Create model using factory
    model = create_piecewise_activation_model()
    
    # Create synthetic data
    n_cells = 10
    n_genes = 5
    u_obs = torch.poisson(torch.ones(n_cells, n_genes) * 50.0)
    s_obs = torch.poisson(torch.ones(n_cells, n_genes) * 100.0)
    
    # This should work without errors if plate names are handled correctly
    try:
        # Test prior predictive sampling
        prior_predictive = Predictive(
            model.forward, 
            guide=None, 
            num_samples=3
        )
        prior_samples = prior_predictive(
            u_obs=None, 
            s_obs=None, 
            num_cells=n_cells,  # Changed from n_cells to num_cells
            num_genes=n_genes   # Changed from n_genes to num_genes
        )
        
        # Test that we got samples for key parameters
        assert "R_on" in prior_samples
        assert "t_star" in prior_samples
        assert prior_samples["R_on"].shape[0] == 3  # num_samples
        
    except RuntimeError as e:
        if "Multiple sample sites named" in str(e):
            pytest.fail(f"Plate name conflict detected: {e}")
        else:
            raise


def test_plate_context_detection():
    """Test detecting if we're already in a plate context."""
    pyro.clear_param_store()
    
    # Test case 1: Not in any plate
    assert not any(frame.name == "cells" for frame in pyro.poutine.runtime._PYRO_STACK)
    
    # Test case 2: Inside a plate
    with pyro.plate("cells", 10, dim=-2):
        plate_frames = [frame for frame in pyro.poutine.runtime._PYRO_STACK 
                       if hasattr(frame, 'name') and frame.name == "cells"]
        assert len(plate_frames) > 0
        
        # Test case 3: Nested plates should fail
        with pytest.raises(RuntimeError, match="Multiple sample sites named 'cells'"):
            with pyro.plate("cells", 10, dim=-2):
                pass


def test_plate_dimension_management():
    """Test that plate dimensions are managed correctly."""
    pyro.clear_param_store()
    
    n_cells = 10
    n_genes = 5
    
    # Create a simple model that uses plates
    def model():
        with pyro.plate("cells", n_cells, dim=-2):
            with pyro.plate("genes", n_genes, dim=-1):
                pyro.sample("x", dist.Normal(0, 1))
        return None
    
    # Test with Predictive - should handle dimensions correctly
    predictive = Predictive(model, guide=None, num_samples=3)
    samples = predictive()
    
    # Check dimensions: [num_samples, cells, genes]
    assert samples["x"].shape == (3, n_cells, n_genes)


if __name__ == "__main__":
    test_plate_name_consistency_in_predictive()
    test_plate_context_detection()
    test_plate_dimension_management()
    print("All tests passed!")