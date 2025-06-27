Feature: Dynamics Model
  As a computational biologist
  I want to model RNA velocity dynamics
  So that I can understand transcriptional dynamics in single-cell data

  Background:
    Given I have a dynamics model component
    And I have input data with unspliced and spliced counts

  Scenario Outline: Piecewise activation dynamics model computes expected counts
    Given I have a PiecewiseActivationDynamicsModel
    When I run the forward method with alpha_off <alpha_off> and gamma_star <gamma_star>
    Then the model should compute expected unspliced and spliced counts
    And the expected counts should follow RNA velocity dynamics

    Examples:
      | alpha_off | gamma_star |
      | 0.3       | 0.8        |

  Scenario Outline: Piecewise activation dynamics model computes steady state
    Given I have a PiecewiseActivationDynamicsModel
    When I compute the steady state with alpha_off <alpha_off> and gamma_star <gamma_star>
    Then the steady state should be computed correctly
    And the steady state values should be positive

    Examples:
      | alpha_off | gamma_star |
      | 0.3       | 0.8        |

  Scenario: Dynamics model handles edge cases
    Given I have a PiecewiseActivationDynamicsModel
    When I run the forward method with edge case parameters
    Then the model should handle the edge case gracefully
    And should not produce NaN or infinite values

  Scenario: Dynamics model with library size correction
    Given I have a PiecewiseActivationDynamicsModel with library size correction
    When I run the forward method with library size factors
    Then the expected counts should be scaled by the library size factors
    And the scaling should be applied correctly
