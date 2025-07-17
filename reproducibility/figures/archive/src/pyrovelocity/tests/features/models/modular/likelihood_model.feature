Feature: Likelihood Model
  As a computational biologist
  I want to define likelihood distributions for RNA velocity observations
  So that I can relate expected counts to observed counts in single-cell data

  Background:
    Given I have a likelihood model component
    And I have input data with unspliced and spliced counts
    And I have expected unspliced and spliced counts

  Scenario: Piecewise activation Poisson likelihood model defines distributions
    Given I have a PiecewiseActivationPoissonLikelihoodModel
    When I run the forward method
    Then the model should define Poisson distributions for unspliced and spliced counts
    And the distributions should use the expected counts as rate parameters
    And the model should register observations with Pyro

  Scenario: Likelihood model with scaling factors
    Given I have a PiecewiseActivationPoissonLikelihoodModel
    When I run the forward method with scaling factors
    Then the distributions should incorporate the scaling factors
    And the rate parameters should be adjusted accordingly

  Scenario: Likelihood model with zero counts
    Given I have a PiecewiseActivationPoissonLikelihoodModel
    And I have data with zero counts
    When I run the forward method
    Then the model should handle zero counts gracefully
    And should not produce errors or warnings

  Scenario: Likelihood model with Pyro plates
    Given I have a PiecewiseActivationPoissonLikelihoodModel
    When I run the forward method with a plate context
    Then the model should use the plate for batch dimensions
    And the observations should be registered with the correct dimensions
