"""
Constants for PyroVelocity modular implementation.

This module contains constants used throughout the PyroVelocity modular implementation,
including plate dimensions for Pyro's plate context.
"""

# Plate dimensions
# These constants define the dimensions used for plates in Pyro models
# Using consistent dimensions ensures proper broadcasting and tensor shapes
GENES_DIM = -1  # Always use -1 for genes
CELLS_DIM = -2  # Always use -2 for cells
BATCH_DIM = -3  # Use -3 for batch if needed

# Parameter shape constants
# These constants define the expected shapes for different types of parameters
# Gene-specific parameters: [num_samples, 1, n_genes]
# Cell-specific parameters: [num_samples, n_cells, 1]
# Cell-gene interactions: [num_samples, n_cells, n_genes]
