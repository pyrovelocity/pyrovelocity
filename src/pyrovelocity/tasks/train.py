import json
import os
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import polars as pl
from anndata._core.anndata import AnnData
from beartype import beartype
from matplotlib.ticker import ScalarFormatter, SymmetricalLogLocator
from mlflow import MlflowClient
from numpy import ndarray
from scvi.model._utils import parse_device_args

from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.logging import configure_logging
from pyrovelocity.models import PyroVelocity
from pyrovelocity.random_state import set_seed
from pyrovelocity.utils import (
    anndata_string,
    load_anndata_from_path,
    print_anndata,
    print_string_diff,
)

logger = configure_logging(__name__)


@beartype
def train_dataset(
    adata: str | Path | AnnData,
    data_set_name: str = "simulated",
    model_identifier: str = "model2",
    models_path: str | Path = "models",
    guide_type: str = "auto",
    model_type: str = "auto",
    batch_size: int = -1,
    use_gpu: str = "auto",
    likelihood: str = "Poisson",
    num_samples: int = 30,
    log_every: int = 100,
    patient_improve: float = 1e-4,
    patient_init: int = 45,
    random_seed: int = 99,
    learning_rate: float = 0.01,
    max_epochs: int = 3000,
    include_prior: bool = True,
    library_size: bool = True,
    offset: bool = True,
    input_type: str = "raw",
    cell_specific_kinetics: Optional[str] = None,
    kinetics_num: int = 2,
    force: bool = False,
) -> Tuple[str, str, Path, Path, Path, Path, Path, Path, Path]:
    """
    Loads processed data, trains model, and saves model and posterior samples.

    Inputs

        - data/processed/{data_set_name}_processed.h5ad

    Outputs

        models/{data_model}/
        ├── trained.h5ad
        ├── posterior_samples.pkl.zst
        └── model/
            ├── attr.pkl
            ├── model_params.pt
            ├── param_store_test.pt
            └── var_names.csv

    The file `models/{data_model}/pyrovelocity.pkl` used in
    `pyrovelocity.tasks.summarize.summarize_dataset` is produced by
    `pyrovelocity.tasks.postprocess.postprocess_dataset`.


    Args:
        adata (str | AnnData):
            Path to a file that can be read to an AnnData object or an AnnData object.
        data_set_name (str, optional): Name of the dataset. Default is "simulated".
        model_identifier (str, optional): Identifier for the model. Default is "model2".
        models_path (str | Path, optional): Path to the models directory. Default is "models".
        guide_type (str, optional):
            The type of guide function for the Pyro model. Default is "auto".
        model_type (str, optional): The type of Pyro model. Default is "auto".
        batch_size (int, optional):
            Batch size for training. Default is -1, which indicates using the full dataset.
        use_gpu (int, optional):
            Whether to use GPU for training. Default is 0, which indicates not using GPU.
        likelihood (str, optional):
            Likelihood function for the Pyro model. Default is "Poisson".
        num_samples (int, optional): Number of posterior samples. Default is 30.
        log_every (int, optional): Frequency of logging progress. Default is 100.
        patient_improve (float, optional):
            Minimum improvement in training loss for early stopping. Default is 5e-4.
        patient_init (int, optional):
            Number of initial training epochs before early stopping is enabled. Default is 30.
        random_seed (int, optional): Random seed for reproducibility. Default is 99.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 0.01.
        max_epochs (int, optional): Maximum number of training epochs. Default is 3000.
        include_prior (bool, optional):
            Whether to include prior information in the model. Default is True.
        library_size (bool, optional): Whether to correct for library size. Default is True.
        offset (bool, optional): Whether to add an offset to the model. Default is True.
        input_type (str, optional): Type of input data. Default is "raw".
        cell_specific_kinetics (Optional[str], optional):
            Name of the attribute containing cell-specific kinetics information. Default is None.
        kinetics_num (int, optional): Number of kinetics parameters. Default is 2.
        force (bool, optional): Whether to overwrite existing files. Default is False.

    Returns:
        Tuple[str, str, Path, Path, Path, Path, Path, Path]:
            Tuple containing the name of the data model, the path to the model
            directory, the path to the trained data, the path to the model, the
            path to the posterior samples, the path to the metrics, the path to
            the run info, and the path to the loss plot.

    Examples:
        >>> train_dataset() # xdoctest: +SKIP
    """
    set_seed(random_seed)
    logger.info(f"Reset random state from seed: {random_seed}")

    # load data
    data_model = f"{data_set_name}_{model_identifier}"
    data_model_path = Path(f"{models_path}/{data_model}")

    trained_data_path = data_model_path / "trained.h5ad"
    model_path = data_model_path / "model"
    posterior_samples_path = data_model_path / "posterior_samples.pkl.zst"
    metrics_path = data_model_path / "metrics.json"
    run_info_path = data_model_path / "run_info.json"
    loss_plot_path = data_model_path / "ELBO.png"
    loss_csv_path = loss_plot_path.with_suffix(".csv")

    logger.info(f"\n\nTraining: {data_model}\n\n")

    logger.info(
        f"\n\nVerifying existence of paths for:\n\n"
        f"  model data: {data_model_path}\n"
    )
    Path(data_model_path).mkdir(parents=True, exist_ok=True)

    # train model
    _accelerator, _devices, device = parse_device_args(
        accelerator=use_gpu, return_device="torch"
    )
    logger.info(
        f"Accelerator type specified as {use_gpu} resolves to:\n"
        f"\taccelerator: {_accelerator}\n"
        f"\tdevices: {_devices}\n"
        f"\tdevice: {device}\n"
    )

    if (
        os.path.isfile(trained_data_path)
        and os.path.exists(model_path)
        and os.path.isfile(posterior_samples_path)
        and not force
    ):
        logger.info(
            f"\n{trained_data_path}\n"
            f"{model_path}\n"
            f"{posterior_samples_path}\n"
            "all exist, set `force=True` to overwrite."
        )
        return (
            data_model,
            str(data_model_path),
            trained_data_path,
            model_path,
            posterior_samples_path,
            metrics_path,
            run_info_path,
            loss_plot_path,
            loss_csv_path,
        )
    else:
        logger.info(f"Training model: {data_model}")

        with mlflow.start_run(
            run_name=f"{data_model}-{uuid.uuid4().hex[:7]}",
            nested=True,
        ) as run:
            mlflow.set_tag(
                "mlflow.runName", f"{data_model}-{run.info.run_id[:7]}"
            )
            print(f"Active run_id: {run.info.run_id}")
            mlflow.log_params(
                {
                    "data_set_name": data_set_name,
                    "model_identifier": model_identifier,
                    "guide_type": guide_type,
                    "model_type": model_type,
                    "batch_size": batch_size,
                }
            )

            adata, trained_model, posterior_samples = train_model(
                adata=adata,
                guide_type=guide_type,
                model_type=model_type,
                batch_size=batch_size,
                use_gpu=use_gpu,
                likelihood=likelihood,
                num_samples=num_samples,
                log_every=log_every,
                patient_improve=patient_improve,
                patient_init=patient_init,
                random_seed=random_seed,
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                include_prior=include_prior,
                library_size=library_size,
                offset=offset,
                input_type=input_type,
                cell_specific_kinetics=cell_specific_kinetics,
                kinetics_num=kinetics_num,
                loss_plot_path=loss_plot_path,
                loss_csv_path=loss_csv_path,
            )

            logger.info("Data attributes after model training")

            run_id = run.info.run_id

        logger.info(f"Saving posterior samples: {posterior_samples_path}\n")
        CompressedPickle.save(
            posterior_samples_path,
            posterior_samples,
        )

        # save metrics
        r = mlflow.get_run(run_id)

        Path(metrics_path).write_text(json.dumps(r.data.metrics, indent=4))

        Path(run_info_path).write_text(
            json.dumps(r.to_dictionary()["info"], indent=4)
        )

        log_run_info(r)

        # postprocess
        if "pancreas" in data_model:
            logger.info("checking shared time")

            def check_shared_time(posterior_samples, adata):
                adata.obs["cell_time"] = (
                    posterior_samples["cell_time"].squeeze().mean(0)
                )
                adata.obs["1-Cytotrace"] = 1 - adata.obs["cytotrace"]

            # Capture state before check_shared_time
            pre_shared_time_representation = anndata_string(adata)

            check_shared_time(posterior_samples, adata)

            # Track changes after check_shared_time
            post_shared_time_representation = anndata_string(adata)
            print_string_diff(
                text1=pre_shared_time_representation,
                text2=post_shared_time_representation,
                diff_title="Check shared time diff",
            )

        print_anndata(adata)

        # save checkpoints
        logger.info(f"Saving trained data: {trained_data_path}")
        adata.write(trained_data_path)

        logger.info(f"Saving model: {model_path}")
        trained_model.save_model(model_path, overwrite=True)

        logger.info(
            f"\nReturning paths to saved data:\n\n"
            f"{data_model_path}\n"
            f"{trained_data_path}\n"
            f"{model_path}\n"
            f"{posterior_samples_path}\n"
            f"{metrics_path}\n"
            f"{run_info_path}\n"
            f"{loss_plot_path}\n"
            f"{loss_csv_path}\n"
        )
        return (
            data_model,
            str(data_model_path),
            trained_data_path,
            model_path,
            posterior_samples_path,
            metrics_path,
            run_info_path,
            loss_plot_path,
            loss_csv_path,
        )


@beartype
def train_model(
    adata: str | Path | AnnData,
    guide_type: str = "auto",
    model_type: str = "auto",
    batch_size: int = -1,
    use_gpu: str = "auto",
    likelihood: str = "Poisson",
    num_samples: int = 30,
    log_every: int = 100,
    patient_improve: float = 1e-4,
    patient_init: int = 45,
    random_seed: int = 99,
    learning_rate: float = 0.01,
    max_epochs: int = 3000,
    include_prior: bool = True,
    library_size: bool = True,
    offset: bool = True,
    input_type: str = "raw",
    cell_specific_kinetics: Optional[str] = None,
    kinetics_num: int = 2,
    loss_plot_path: str | Path = "loss_plot.png",
    loss_csv_path: str | Path = "loss_plot.png.csv",
) -> Tuple[AnnData, PyroVelocity, Dict[str, ndarray]]:
    """
    Train a PyroVelocity model to provide probabilistic estimates of RNA velocity
    for single-cell RNA sequencing data with quantified splice variants.

    These arguments are deprecated

    # svi_train: bool = False,
    # train_size: float = 1.0,
    # cell_state: str = "clusters",
    # svi_train (bool, optional): Whether to use Stochastic Variational Inference for training. Default is False.
    # train_size (float, optional): Proportion of data to be used for training. Default is 1.0.
    # cell_state (str, optional): Cell state attribute in the AnnData object. Default is "clusters".

    Args:
        adata (str | AnnData): Path to a file that can be read to an AnnData object or an AnnData object.
        guide_type (str, optional): The type of guide function for the Pyro model. Default is "auto".
        model_type (str, optional): The type of Pyro model. Default is "auto".
        batch_size (int, optional): Batch size for training. Default is -1, which indicates using the full dataset.
        use_gpu (int, optional): Whether to use GPU for training. Default is 0, which indicates not using GPU.
        likelihood (str, optional): Likelihood function for the Pyro model. Default is "Poisson".
        num_samples (int, optional): Number of posterior samples. Default is 30.
        log_every (int, optional): Frequency of logging progress. Default is 100.
        patient_improve (float, optional): Minimum improvement in training loss for early stopping. Default is 5e-4.
        patient_init (int, optional): Number of initial training epochs before early stopping is enabled. Default is 30.
        random_seed (int, optional): Random seed for reproducibility. Default is 99.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 0.01.
        max_epochs (int, optional): Maximum number of training epochs. Default is 3000.
        include_prior (bool, optional): Whether to include prior information in the model. Default is True.
        library_size (bool, optional): Whether to correct for library size. Default is True.
        offset (bool, optional): Whether to add an offset to the model. Default is True.
        input_type (str, optional): Type of input data. Default is "raw".
        cell_specific_kinetics (Optional[str], optional): Name of the attribute containing cell-specific kinetics information. Default is None.
        kinetics_num (int, optional): Number of kinetics parameters. Default is 2.
        loss_plot_path (str, optional): Path to save the loss plot. Default is "loss_plot.png".
        loss_csv_path (str, optional): Path to save the loss data as CSV. Default is "loss_plot.png.csv".

    Returns:
        Tuple[AnnData, PyroVelocity, Dict[str, ndarray]]: A tuple containing the processed AnnData object, the trained PyroVelocity model, and a dictionary of posterior samples.

    Examples:
        >>> # xdoctest: +SKIP
        >>> from pyrovelocity.tasks.train import train_model
        >>> from pyrovelocity.utils import generate_sample_data
        >>> from pyrovelocity.tasks.preprocess import copy_raw_counts
        >>> from pathlib import Path
        >>> tmpdir = None
        >>> try:
        >>>     tmp = getfixture("tmp_path")
        >>> except NameError:
        >>>     import tempfile
        >>>     tmpdir = tempfile.TemporaryDirectory()
        >>>     tmp = tmpdir.name
        >>> loss_plot_path = str(tmp) + "/loss_plot_docs.png"
        >>> loss_csv_path = str(tmp) + "/loss_plot_docs.png.csv"
        >>> print(loss_plot_path)
        >>> adata = generate_sample_data(random_seed=99)
        >>> copy_raw_counts(adata)
        >>> _, model, posterior_samples = train_model(
        ...   adata,
        ...   use_gpu="auto",
        ...   random_seed=99,
        ...   max_epochs=200,
        ...   loss_plot_path=loss_plot_path,
        ...   loss_csv_path=loss_csv_path,
        ... )
    """
    set_seed(random_seed)
    logger.info(f"Reset random state from seed: {random_seed}")

    if isinstance(adata, str | Path):
        adata = load_anndata_from_path(adata)

    logger.info(f"AnnData object prior to model training")
    pre_setup_representation = anndata_string(adata)
    print(pre_setup_representation)

    PyroVelocity.setup_anndata(adata)

    # Track changes after setup_anndata
    post_setup_representation = anndata_string(adata)
    print_string_diff(
        text1=pre_setup_representation,
        text2=post_setup_representation,
        diff_title="PyroVelocity.setup_anndata diff",
    )

    model = PyroVelocity(
        adata,
        likelihood=likelihood,
        model_type=model_type,
        guide_type=guide_type,
        correct_library_size=library_size,
        add_offset=offset,
        include_prior=include_prior,
        input_type=input_type,
        cell_specific_kinetics=cell_specific_kinetics,
        kinetics_num=kinetics_num,
    )

    if batch_size == -1:
        batch_size = adata.shape[0]

    if batch_size >= adata.shape[0]:
        losses = model.train_faster(
            max_epochs=max_epochs,
            lr=learning_rate,
            use_gpu=use_gpu,
            seed=random_seed,
            patient_improve=patient_improve,
            patient_init=patient_init,
            log_every=log_every,
        )
    else:
        losses = model.train_faster_with_batch(
            max_epochs=max_epochs,
            batch_size=batch_size,
            log_every=log_every,
            lr=learning_rate,
            use_gpu=use_gpu,
            seed=random_seed,
            patient_improve=patient_improve,
            patient_init=patient_init,
        )
    fig, ax = plt.subplots()
    fig.set_size_inches(2.5, 1.5)

    numpy_losses_range = np.arange(len(losses))
    numpy_negative_elbo = -np.array(losses)

    df = pl.DataFrame(
        {
            "Epoch": numpy_losses_range,
            "-ELBO": numpy_negative_elbo,
        }
    )
    df.write_csv(loss_csv_path)

    ax.scatter(
        numpy_losses_range,
        numpy_negative_elbo,
        label="train",
        alpha=0.25,
    )
    set_loss_plot_axes(ax)

    # Capture state before posterior samples generation
    pre_posterior_samples_representation = anndata_string(adata)

    posterior_samples = model.generate_posterior_samples(
        model.adata, num_samples=num_samples, batch_size=512
    )

    # Track changes after posterior samples generation
    post_posterior_samples_representation = anndata_string(adata)
    print_string_diff(
        text1=pre_posterior_samples_representation,
        text2=post_posterior_samples_representation,
        diff_title="Posterior samples generation diff",
    )

    fig.savefig(loss_plot_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    logger.info(f"AnnData object after model training")
    print_anndata(adata)
    return (
        adata,
        model,
        posterior_samples,
    )


def set_loss_plot_axes(ax):
    ax.set_yscale("symlog")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("-ELBO")

    locator = SymmetricalLogLocator(base=10, linthresh=1, subs=[1.0])
    ax.yaxis.set_major_locator(locator)

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)


def log_run_info(r: mlflow.entities.run.Run) -> None:
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [
        f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")
    ]
    logger.info(
        f"\nrun_id: {r.info.run_id}\n"
        f"artifacts: {artifacts}\n"
        f"params: {r.data.params}\n"
        f"metrics: {r.data.metrics}\n"
        f"tags: {tags}\n"
    )
