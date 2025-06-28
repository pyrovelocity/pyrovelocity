Feature: JAX Prior Model
  As a computational biologist
  I want to specify prior distributions using NumPyro
  So that I can incorporate prior knowledge in JAX-based RNA velocity modeling

  Background:
    Given I have a JAX prior model component
    And I have a JAX PRNG key

  Scenario: JAX prior model samples parameters
    Given I have a JAX PiecewiseActivationPriorModel
    When I run the forward method with NumPyro sampling
    Then the model should sample prior parameters using NumPyro distributions
    And the sampled parameters should be JAX arrays
    And the parameters should follow the specified prior distributions

  Scenario: JAX prior model with deterministic parameters
    Given I have a JAX PiecewiseActivationPriorModel
    When I run the forward method with deterministic mode
    Then the model should use deterministic parameter values
    And the parameters should be JAX arrays with correct shapes
    And the parameters should be within valid ranges

  Scenario: JAX prior model parameter hierarchy
    Given I have a JAX PiecewiseActivationPriorModel
    When I sample parameters with hierarchical structure
    Then the model should respect parameter hierarchy
    And shared parameters should be properly broadcast
    And the parameter shapes should be consistent across genes and cells

  Scenario: JAX prior model with PRNG key splitting
    Given I have a JAX PiecewiseActivationPriorModel
    When I sample parameters with PRNG key splitting
    Then each parameter should use a unique PRNG subkey
    And the sampling should be reproducible with the same root key
    And the PRNG state should be properly managed

  Scenario: JAX prior model gamma boundary handling
    Given I have a JAX PiecewiseActivationPriorModel
    When I sample gamma_star parameters near 1.0
    Then the model should handle the boundary case appropriately
    And the sampled values should maintain numerical stability
    And the prior should prevent problematic gamma values