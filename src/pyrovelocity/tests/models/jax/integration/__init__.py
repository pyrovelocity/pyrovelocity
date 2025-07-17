"""
Integration tests for JAX models using pytest-bdd.

This package contains step definitions for behavioral testing of the JAX implementation.

Note that when using pytest-bdd, it is important to ensure that the feature file is correctly imported and that the step definitions are properly defined.
Here is a brief sketch of how to use pytest-bdd importing the feature files and referencing the step definitions in test functions:

```python
from importlib.resources import files
from pytest_bdd import given, parsers, scenarios, then, when

# Import the feature file using importlib.resources
scenarios(
    str(
        files("pyrovelocity.tests.features")
        / "validation"
        / "parameter_recovery"
        / "parameter_generation.feature"
    )
)

@given("I have a ParameterGenerator component", target_fixture="parameter_generator")
def parameter_generator_component():
    pass


@given(parsers.parse("a model with {prior_type} prior"))
def model_with_prior(parameter_generator, prior_type):
    pass


@when(parsers.parse("I generate {num_parameter_sets:d} parameter sets with {num_genes:d} genes"))
def generate_parameter_sets(parameter_generator, num_parameter_sets, num_genes):
    pass


@then(parsers.parse("I should get {num_parameter_sets:d} unique parameter sets"))
def check_num_parameter_sets(parameter_generator, num_parameter_sets):
    pass
```
"""
