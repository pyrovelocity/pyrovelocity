Feature: JAX PyroVelocity Model
  As a computational biologist
  I want to use a JAX-based RNA velocity model
  So that I can analyze transcriptional dynamics with high-performance computation

  Background:
    Given I have input data with unspliced and spliced JAX arrays
    And I have a JAX PiecewiseActivationDynamicsModel
    And I have a JAX PiecewiseActivationPriorModel  
    And I have a JAX PiecewiseActivationPoissonLikelihoodModel
    And I have a JAX AutoGuideFactory
    And I have a JAX PRNG key

  Scenario: Creating a JAX PyroVelocity model
    When I create a JAX PyroVelocity model with these components
    Then the model should be properly initialized
    And the model should have the correct component structure
    And the model should implement the NumPyro forward method
    And all components should work with JAX arrays

  Scenario: Running the JAX model forward method
    Given I have created a JAX PyroVelocity model
    When I run the forward method with JAX arrays
    Then the model should process the data through all components
    And the output should include expected counts as JAX arrays
    And the model should register all parameters and observations with NumPyro
    And the computation should be fully differentiable

  Scenario: JAX model JIT compilation
    Given I have created a JAX PyroVelocity model
    When I JIT compile the model
    Then the model should compile successfully
    And the compiled model should execute efficiently
    And the outputs should match the non-compiled version

  Scenario: JAX model inference with NumPyro SVI
    Given I have created a JAX PyroVelocity model
    When I run NumPyro SVI for 100 steps
    Then the inference should converge
    And the loss should decrease over iterations
    And the variational parameters should be updated
    And all computations should use JAX operations

  Scenario: JAX model posterior sampling
    Given I have a trained JAX PyroVelocity model
    When I generate posterior samples using NumPyro MCMC
    Then the samples should be JAX arrays
    And the samples should include all model parameters
    And the sampling should be reproducible with the same PRNG key

  Scenario: JAX model velocity computation
    Given I have a trained JAX PyroVelocity model with posterior samples
    When I compute RNA velocity using JAX operations
    Then the velocity vectors should be computed for each cell
    And the velocity should be JAX arrays
    And the computation should be vectorized and efficient

  Scenario: JAX model with gamma boundary cases
    Given I have created a JAX PyroVelocity model
    When I run inference with gamma_star values near 1.0
    Then the model should handle the boundary case gracefully
    And the inference should remain numerically stable
    And the results should be physically meaningful