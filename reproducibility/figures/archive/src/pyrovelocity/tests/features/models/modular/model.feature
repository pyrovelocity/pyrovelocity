Feature: PyroVelocity Model
  As a computational biologist
  I want to use a modular RNA velocity model
  So that I can analyze transcriptional dynamics in single-cell data

  Background:
    Given I have input data with unspliced and spliced counts
    And I have a PiecewiseActivationDynamicsModel
    And I have a PiecewiseActivationPriorModel
    And I have a PiecewiseActivationPoissonLikelihoodModel
    And I have an AutoGuideFactory
    And I have a trained PyroVelocity model
    And I have a trained PyroVelocity model with posterior samples
    And I have a trained PyroVelocity model with velocity results

  Scenario: Creating a PyroVelocity model
    When I create a PyroVelocity model with these components
    Then the model should be properly initialized
    And the model should have the correct component structure
    And the model should implement the forward method

  Scenario: Running the forward method
    Given I have created a PyroVelocity model
    And I have input data with unspliced and spliced counts
    When I run the forward method
    Then the model should process the data through all components
    And the output should include expected counts and distributions
    And the model should register all parameters and observations with Pyro

  Scenario: Training the model with AnnData
    Given I have created a PyroVelocity model
    And I have an AnnData object with RNA velocity data
    When I train the model for 10 epochs
    Then the model should converge
    And the loss should decrease
    And the posterior samples should be stored

  Scenario: Generating posterior samples
    Given I have a trained PyroVelocity model
    When I generate 100 posterior samples
    Then the samples should have the correct structure
    And the samples should include all model parameters
    And the samples should reflect the posterior distribution

  Scenario: Computing RNA velocity
    Given I have a trained PyroVelocity model with posterior samples
    When I compute RNA velocity
    Then the velocity vectors should be computed for each cell
    And the velocity should reflect the transcriptional dynamics
    And the velocity should be stored in the model state

  Scenario: Storing results in AnnData
    Given I have a trained PyroVelocity model with velocity results
    And I have an AnnData object
    When I store the results in the AnnData object
    Then the AnnData object should contain the velocity results
    And the AnnData object should contain the model parameters
    And the AnnData object should be ready for downstream analysis
