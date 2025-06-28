Feature: JAX Guide Model
  As a computational biologist  
  I want to use variational guides with NumPyro
  So that I can perform efficient inference in JAX-based RNA velocity models

  Background:
    Given I have a JAX guide model component
    And I have a JAX PRNG key

  Scenario: JAX AutoNormal guide creation
    Given I have a JAX AutoGuideFactory with AutoNormal guide type
    When I create a guide for a NumPyro model
    Then the guide should be a NumPyro AutoNormal guide
    And the guide should handle JAX arrays correctly
    And the guide should be compatible with NumPyro SVI

  Scenario: JAX guide parameter initialization
    Given I have a JAX guide model
    When I initialize guide parameters with a PRNG key
    Then the guide parameters should be JAX arrays
    And the parameters should be initialized appropriately
    And the initialization should be reproducible with the same key

  Scenario: JAX guide sampling
    Given I have a JAX guide model
    When I sample from the guide distribution
    Then the samples should be JAX arrays
    And the sampling should use the provided PRNG key
    And the samples should follow the guide distribution

  Scenario: JAX guide log probability computation
    Given I have a JAX guide model
    When I compute the log probability of samples
    Then the log probability should be computed using JAX operations
    And the computation should be differentiable
    And the result should be a JAX array

  Scenario: JAX guide with batch dimensions
    Given I have a JAX guide model
    When I work with batched data
    Then the guide should handle batch dimensions correctly
    And the variational parameters should be properly shaped
    And the sampling should preserve batch structure