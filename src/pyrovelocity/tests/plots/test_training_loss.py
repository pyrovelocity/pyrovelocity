"""Tests for training loss plotting functionality."""

import pytest
import torch
import matplotlib.pyplot as plt
from unittest.mock import Mock, MagicMock
from pyrovelocity.plots.predictive import plot_training_loss


class TestTrainingLossPlot:
    """Test training loss plotting functionality."""

    def test_plot_training_loss_success(self, tmp_path):
        """Test successful training loss plot generation."""
        # Create mock model with training history
        mock_model = Mock()
        mock_model.state = Mock()
        
        # Create mock inference state with training history
        mock_inference_state = Mock()
        mock_training_state = Mock()
        mock_training_state.loss_history = [-100.5, -95.2, -90.1, -88.3, -87.0, -86.5]
        mock_inference_state.training_state = mock_training_state
        
        # Use the new direct access pattern instead of metadata
        mock_model.state.inference_state = mock_inference_state
        
        # Test plot generation
        fig = plot_training_loss(
            model=mock_model,
            save_path=str(tmp_path),
            file_prefix="test",
            moving_average_window=3
        )
        
        # Verify figure was created
        assert isinstance(fig, plt.Figure)
        
        # Verify axes content
        ax = fig.get_axes()[0]
        assert ax.get_xlabel() == 'Epoch'
        assert ax.get_ylabel() == 'ELBO'
        assert ax.get_title() == 'Training Loss (Evidence Lower Bound)'
        
        # Verify legend exists
        legend = ax.get_legend()
        assert legend is not None
        
        # Verify files were saved
        assert (tmp_path / "test_training_loss.png").exists()
        assert (tmp_path / "test_training_loss.pdf").exists()
        
        plt.close(fig)

    def test_plot_training_loss_no_state(self):
        """Test error handling when model has no state."""
        mock_model = Mock()
        mock_model.state = None
        
        with pytest.raises(ValueError, match="Model has no state"):
            plot_training_loss(mock_model)

    def test_plot_training_loss_no_inference_state(self):
        """Test error handling when model has no inference state."""
        mock_model = Mock()
        mock_model.state = Mock()
        mock_model.state.inference_state = None
        
        with pytest.raises(ValueError, match="Model has no inference state"):
            plot_training_loss(mock_model)

    def test_plot_training_loss_no_training_state(self):
        """Test error handling when model has no training state."""
        mock_model = Mock()
        mock_model.state = Mock()
        mock_inference_state = Mock()
        mock_inference_state.training_state = None
        mock_model.state.inference_state = mock_inference_state
        
        with pytest.raises(ValueError, match="Model has no training history"):
            plot_training_loss(mock_model)

    def test_plot_training_loss_empty_history(self):
        """Test error handling when training history is empty."""
        mock_model = Mock()
        mock_model.state = Mock()
        mock_inference_state = Mock()
        mock_training_state = Mock()
        mock_training_state.loss_history = []
        mock_inference_state.training_state = mock_training_state
        mock_model.state.inference_state = mock_inference_state
        
        with pytest.raises(ValueError, match="Model has no training history"):
            plot_training_loss(mock_model)

    def test_plot_training_loss_short_history(self, tmp_path):
        """Test plot generation with short training history (no moving average)."""
        mock_model = Mock()
        mock_model.state = Mock()
        mock_inference_state = Mock()
        mock_training_state = Mock()
        mock_training_state.loss_history = [-100.0, -95.0]  # Only 2 epochs
        mock_inference_state.training_state = mock_training_state
        mock_model.state.inference_state = mock_inference_state
        
        # Test with moving average window larger than history
        fig = plot_training_loss(
            model=mock_model,
            moving_average_window=10  # Larger than history length
        )
        
        # Should still create plot without moving average
        assert isinstance(fig, plt.Figure)
        
        # Verify axes content
        ax = fig.get_axes()[0]
        assert ax.get_xlabel() == 'Epoch'
        assert ax.get_ylabel() == 'ELBO'
        
        plt.close(fig)

    def test_plot_training_loss_elbo_conversion(self, tmp_path):
        """Test that negative loss values are correctly converted to positive ELBO."""
        mock_model = Mock()
        mock_model.state = Mock()
        mock_inference_state = Mock()
        mock_training_state = Mock()
        # Negative loss values (from minimization objective)
        mock_training_state.loss_history = [-100.0, -90.0, -80.0, -70.0]
        mock_inference_state.training_state = mock_training_state
        mock_model.state.inference_state = mock_inference_state
        
        fig = plot_training_loss(model=mock_model, moving_average_window=2)
        
        # Get the plotted data
        ax = fig.get_axes()[0]
        lines = ax.get_lines()
        
        # Find the raw data line (should be first)
        raw_data_line = None
        for line in lines:
            if line.get_alpha() == 0.3:  # Raw data connecting line
                raw_data_line = line
                break
        
        if raw_data_line is not None:
            y_data = raw_data_line.get_ydata()
            # Verify all ELBO values are positive
            assert all(y > 0 for y in y_data)
            # Verify conversion: -(-100) = 100, etc.
            expected_values = [100.0, 90.0, 80.0, 70.0]
            assert list(y_data) == expected_values
        
        plt.close(fig)
