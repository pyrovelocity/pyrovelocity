Feature: JAX Likelihood Model  
  As a computational biologist
  I want to model observation likelihoods using NumPyro
  So that I can fit JAX-based RNA velocity models to data

  Background:
    Given I have a JAX likelihood model component
    And I have input data with unspliced and spliced JAX arrays
    And I have a JAX PRNG key

  Scenario: JAX Poisson likelihood model observes data
    Given I have a JAX PiecewiseActivationPoissonLikelihoodModel
    When I run the forward method with expected counts as JAX arrays
    Then the model should observe data using NumPyro Poisson distributions
    And the likelihood should be computed with JAX operations
    And the observations should be registered with NumPyro

  Scenario: JAX likelihood model with library size correction
    Given I have a JAX PiecewiseActivationPoissonLikelihoodModel
    When I run the forward method with library size factors as JAX arrays
    Then the expected counts should be scaled by library size
    And the scaling should use JAX broadcasting
    And the corrected counts should be used for likelihood computation

  Scenario: JAX likelihood model preprocessing
    Given I have a JAX PiecewiseActivationPoissonLikelihoodModel
    When I preprocess the input data
    Then the preprocessing should use JAX operations
    And the data should be converted to appropriate JAX array types
    And the preprocessing should be differentiable

  Scenario: JAX likelihood model with zero counts
    Given I have a JAX PiecewiseActivationPoissonLikelihoodModel
    When I run the forward method with zero count observations
    Then the model should handle zero counts appropriately
    And the Poisson likelihood should be computed correctly
    And no numerical issues should arise

  Scenario: JAX likelihood model batch processing
    Given I have a JAX PiecewiseActivationPoissonLikelihoodModel
    When I process data in batches using JAX vectorization
    Then the model should handle batched computation efficiently
    And the likelihood computation should be vectorized
    And the batch dimensions should be preserved correctly