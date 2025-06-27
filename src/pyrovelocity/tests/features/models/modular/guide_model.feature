Feature: Inference Guide
  As a computational biologist
  I want to define variational distributions for RNA velocity inference
  So that I can perform efficient approximate Bayesian inference

  Background:
    Given I have an inference guide component
    And I have a PyroVelocity model

  Scenario: AutoGuide factory creates guide
    Given I have an AutoGuideFactory with guide_type="AutoNormal"
    When I create a guide for the model
    Then the guide should be an instance of AutoNormal
    And the guide should be compatible with the model
    And the guide should have the correct parameter structure

  Scenario: AutoGuide factory with blocking parameters
    Given I have an AutoGuideFactory with parameter blocking
    When I create a guide for the model
    Then the guide should block parameters correctly
    And the guide should handle parameter constraints properly

  Scenario: Guide with parameter initialization
    Given I have an AutoGuideFactory with init_scale=0.1
    When I create a guide for the model
    Then the guide should initialize parameters with the specified scale
    And the initialization should affect the initial variational distribution

  Scenario: Guide with different variational families
    Given I have an AutoGuideFactory with guide_type="<guide_type>"
    When I create a guide for the model
    Then the guide should be an instance of <guide_class>
    And the guide should have the appropriate variational family

    Examples:
      | guide_type           | guide_class                  |
      | AutoNormal           | AutoNormal                   |
      | AutoDiagonalNormal   | AutoDiagonalNormal           |
      | AutoLowRankMultivariateNormal | AutoLowRankMultivariateNormal |
      | AutoDelta            | AutoDelta                    |

  Scenario: Guide with custom initialization
    Given I have an AutoGuideFactory with custom initialization
    When I create a guide for the model
    Then the guide should use the custom initialization
    And the initial variational parameters should reflect the custom values
