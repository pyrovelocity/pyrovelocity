"""
Parameter metadata definitions for PyroVelocity modular components.

This module contains parameter metadata definitions for different model components,
providing standardized information about parameter semantics, display formatting,
and biological interpretations.
"""

from typing import Dict

from pyrovelocity.models.modular.interfaces import (
    ComponentParameterMetadata,
    ParameterMetadata,
)


def create_piecewise_activation_prior_metadata() -> ComponentParameterMetadata:
    """
    Create parameter metadata for the piecewise activation prior model.
    
    This function defines metadata for all parameters in the piecewise activation
    prior model, based on the mathematical descriptions and biological interpretations
    from the validation study documentation.
    
    Returns:
        ComponentParameterMetadata for the piecewise activation prior model.
    """
    parameters = {
        # Hierarchical time parameters
        "T_M_star": ParameterMetadata(
            name="T_M_star",
            display_name=r"$T^*_M$",
            short_label="Max Time",
            description="Maximum dimensionless time coordinate for the population",
            units="dimensionless time",
            typical_range=(1.0, 10.0),
            biological_interpretation="Overall duration of the biological process being modeled",
            plot_order=1
        ),
        
        "t_loc": ParameterMetadata(
            name="t_loc",
            display_name=r"$t_{loc}$",
            short_label="Time Loc",
            description="Location parameter for hierarchical time distribution",
            units="dimensionless",
            typical_range=(0.1, 2.0),
            biological_interpretation="Central tendency of cell time coordinates in the population",
            plot_order=2
        ),
        
        "t_scale": ParameterMetadata(
            name="t_scale",
            display_name=r"$t_{scl}$",
            short_label="Time Scale",
            description="Scale parameter for hierarchical time distribution",
            units="dimensionless",
            typical_range=(0.1, 1.0),
            biological_interpretation="Spread of cell time coordinates around the population mean",
            plot_order=3
        ),

        "tilde_t": ParameterMetadata(
            name="tilde_t",
            display_name=r"$\tilde{t}$",
            short_label="Norm Time",
            description="Normalized cell time coordinates before scaling",
            units="dimensionless",
            typical_range=(-2.0, 4.0),
            biological_interpretation="Cell-specific time coordinates sampled from hierarchical Normal distribution",
            plot_order=4
        ),

        "t_star": ParameterMetadata(
            name="t_star",
            display_name=r"$t^*$",
            short_label="Cell Time",
            description="Individual cell latent time coordinates",
            units="dimensionless time",
            typical_range=(0.0, 10.0),
            biological_interpretation="Progression of individual cells through the biological process",
            plot_order=5
        ),

        "t_star_normalized": ParameterMetadata(
            name="t_star_normalized",
            display_name=r"$t^*_N$",
            short_label="Normalized Cell Time",
            description="Normalized cell time coordinates in [0,1] interval",
            units="dimensionless fraction",
            typical_range=(0.0, 1.0),
            biological_interpretation="Relative progression of individual cells through the biological process (0=start, 1=end)",
            plot_order=6
        ),
        
        # Piecewise activation parameters (corrected parameterization)
        # Note: alpha_off is fixed at 1.0 (not inferred), alpha_on computed from R_on
        "R_on": ParameterMetadata(
            name="R_on",
            display_name=r"$R_{on}$",
            short_label="Fold Change",
            description="Fold-change in transcription rate during activation (R_on = α*_on/α*_off)",
            units="dimensionless ratio",
            typical_range=(1.5, 4.2),
            biological_interpretation="Magnitude of transcriptional upregulation during gene activation phase",
            plot_order=7
        ),

        "gamma_star": ParameterMetadata(
            name="gamma_star",
            display_name=r"$\gamma^*$",
            short_label="Decay Rate",
            description="Relative degradation rate (splicing vs degradation balance)",
            units="dimensionless rate ratio",
            typical_range=(0.3, 3.0),
            biological_interpretation="Balance between mRNA splicing and degradation kinetics; γ*=1 represents balanced kinetics",
            plot_order=8
        ),


        "t_on_star": ParameterMetadata(
            name="t_on_star",
            display_name=r"$t^*_{on}$",
            short_label="Onset Time",
            description="Dimensionless activation onset time (allows negative values for pre-activation)",
            units="dimensionless time",
            typical_range=(-1.0, 1.4),
            biological_interpretation="When during the process each gene begins its activation phase; negative values indicate pre-activation",
            plot_order=9
        ),

        "delta_star": ParameterMetadata(
            name="delta_star",
            display_name=r"$\delta^*$",
            short_label="Duration",
            description="Dimensionless activation duration",
            units="dimensionless time",
            typical_range=(0.1, 1.0),
            biological_interpretation="How long each gene remains in its activated state",
            plot_order=10
        ),
        
        # Observation model parameters
        "U_0i": ParameterMetadata(
            name="U_0i",
            display_name=r"$U_{0i}$",
            short_label="Scale Factor",
            description="Gene-specific scaling factor for observation model",
            units="count scale",
            typical_range=(10.0, 1000.0),
            biological_interpretation="Gene-specific expression scale, accounting for differences in gene expression levels",
            plot_order=11
        ),

        "lambda_j": ParameterMetadata(
            name="lambda_j",
            display_name=r"$\lambda_j$",
            short_label="Capture Eff",
            description="Cell-specific capture efficiency",
            units="efficiency ratio",
            typical_range=(0.1, 2.0),
            biological_interpretation="Technical variation in RNA capture and sequencing efficiency across cells",
            plot_order=12
        ),

        # Latent RNA concentrations (true/unobserved values)
        "ut": ParameterMetadata(
            name="ut",
            display_name=r"$u^*_{ij}$",
            short_label="Latent Unspliced",
            description="Latent dimensionless unspliced RNA concentrations",
            units="dimensionless concentration",
            typical_range=(0.0, 10.0),
            biological_interpretation="True underlying unspliced RNA concentrations before observation noise and technical effects",
            plot_order=13
        ),

        "st": ParameterMetadata(
            name="st",
            display_name=r"$s^*_{ij}$",
            short_label="Latent Spliced",
            description="Latent dimensionless spliced RNA concentrations",
            units="dimensionless concentration",
            typical_range=(0.0, 10.0),
            biological_interpretation="True underlying spliced RNA concentrations before observation noise and technical effects",
            plot_order=14
        ),

        # Observed RNA counts (measured values)
        "u_obs": ParameterMetadata(
            name="u_obs",
            display_name=r"$u_{ij}$",
            short_label="Observed Unspliced",
            description="Observed unspliced RNA counts from sequencing",
            units="RNA counts",
            typical_range=(0.0, 1000.0),
            biological_interpretation="Measured unspliced RNA counts including technical noise, capture efficiency, and sequencing depth effects",
            plot_order=15
        ),

        "s_obs": ParameterMetadata(
            name="s_obs",
            display_name=r"$s_{ij}$",
            short_label="Observed Spliced",
            description="Observed spliced RNA counts from sequencing",
            units="RNA counts",
            typical_range=(0.0, 1000.0),
            biological_interpretation="Measured spliced RNA counts including technical noise, capture efficiency, and sequencing depth effects",
            plot_order=16
        ),
    }
    
    return ComponentParameterMetadata(
        component_name="piecewise_activation_prior",
        component_type="prior",
        parameters=parameters,
        description="Prior distributions for piecewise activation model with hierarchical time structure"
    )


def create_lognormal_prior_metadata() -> ComponentParameterMetadata:
    """
    Create parameter metadata for the standard log-normal prior model.
    
    Returns:
        ComponentParameterMetadata for the log-normal prior model.
    """
    parameters = {
        "alpha": ParameterMetadata(
            name="alpha",
            display_name=r"$\alpha$",
            short_label="Transcription",
            description="Transcription rate parameter",
            units="rate",
            typical_range=(0.1, 10.0),
            biological_interpretation="Rate of pre-mRNA transcription",
            plot_order=1
        ),
        
        "beta": ParameterMetadata(
            name="beta",
            display_name=r"$\beta$",
            short_label="Splicing",
            description="Splicing rate parameter",
            units="rate",
            typical_range=(0.1, 10.0),
            biological_interpretation="Rate of pre-mRNA splicing to mature mRNA",
            plot_order=2
        ),
        
        "gamma": ParameterMetadata(
            name="gamma",
            display_name=r"$\gamma$",
            short_label="Degradation",
            description="mRNA degradation rate parameter",
            units="rate",
            typical_range=(0.1, 10.0),
            biological_interpretation="Rate of mature mRNA degradation",
            plot_order=3
        ),
        
        "scaling": ParameterMetadata(
            name="scaling",
            display_name=r"$s$",
            short_label="Scaling",
            description="Global scaling factor",
            units="scale factor",
            typical_range=(0.1, 10.0),
            biological_interpretation="Overall expression scale adjustment",
            plot_order=4
        ),
    }
    
    return ComponentParameterMetadata(
        component_name="lognormal_prior",
        component_type="prior",
        parameters=parameters,
        description="Log-normal prior distributions for standard RNA velocity parameters"
    )


# Registry of metadata providers
PARAMETER_METADATA_REGISTRY: Dict[str, ComponentParameterMetadata] = {
    "piecewise_activation_prior": create_piecewise_activation_prior_metadata(),
    "lognormal_prior": create_lognormal_prior_metadata(),
}


def get_parameter_metadata(component_name: str) -> ComponentParameterMetadata:
    """
    Get parameter metadata for a named component.
    
    Args:
        component_name: Name of the component to get metadata for
        
    Returns:
        ComponentParameterMetadata for the component, or None if not found
        
    Raises:
        KeyError: If component_name is not found in the registry
    """
    if component_name not in PARAMETER_METADATA_REGISTRY:
        raise KeyError(f"No parameter metadata found for component '{component_name}'")
    
    return PARAMETER_METADATA_REGISTRY[component_name]


def get_parameter_short_labels(component_name: str) -> Dict[str, str]:
    """
    Get short labels for parameters of a named component.
    
    Args:
        component_name: Name of the component
        
    Returns:
        Dictionary mapping parameter names to short labels
    """
    try:
        metadata = get_parameter_metadata(component_name)
        return metadata.get_short_labels()
    except KeyError:
        return {}


def get_parameter_display_names(component_name: str) -> Dict[str, str]:
    """
    Get LaTeX display names for parameters of a named component.
    
    Args:
        component_name: Name of the component
        
    Returns:
        Dictionary mapping parameter names to LaTeX display names
    """
    try:
        metadata = get_parameter_metadata(component_name)
        return metadata.get_display_names()
    except KeyError:
        return {}
