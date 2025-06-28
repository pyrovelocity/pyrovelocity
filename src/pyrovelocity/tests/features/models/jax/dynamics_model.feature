Feature: JAX Dynamics Model
  As a computational biologist
  I want to model RNA velocity dynamics using JAX
  So that I can understand transcriptional dynamics with high-performance computation

  Background:
    Given I have a JAX dynamics model component
    And I have input data with unspliced and spliced JAX arrays
    And I have a JAX PRNG key

  Scenario Outline: JAX piecewise activation dynamics model computes expected counts
    Given I have a JAX PiecewiseActivationDynamicsModel
    When I run the forward method with JAX arrays and gamma_star <gamma_star>
    Then the model should compute expected unspliced and spliced JAX arrays
    And the expected counts should follow RNA velocity dynamics
    And the output arrays should be JAX DeviceArrays

    Examples:
      | gamma_star |
      | 0.8        |
      | 1.0        |
      | 1.2        |

  Scenario: JAX dynamics model handles gamma boundary case
    Given I have a JAX PiecewiseActivationDynamicsModel
    When I run the forward method with gamma_star very close to 1.0
    Then the model should handle the numerical boundary case gracefully
    And should not produce NaN or infinite values
    And the computation should remain numerically stable

  Scenario: JAX dynamics model with JIT compilation
    Given I have a JAX PiecewiseActivationDynamicsModel
    When I JIT compile the forward method
    And I run the compiled method with JAX arrays
    Then the model should execute with JIT acceleration
    And the output should match the non-JIT version
    And the JAX arrays should maintain correct dtypes

  Scenario: JAX PRNG key propagation
    Given I have a JAX PiecewiseActivationDynamicsModel
    When I run the forward method with a specific PRNG key
    Then the model should use the PRNG key correctly
    And subsequent calls with the same key should be deterministic
    And the PRNG key should be properly consumed

  Scenario: JAX dynamics model steady state computation
    Given I have a JAX PiecewiseActivationDynamicsModel
    When I compute the steady state with JAX arrays
    Then the steady state should be computed using JAX operations
    And the steady state values should be positive JAX arrays
    And the computation should be differentiable