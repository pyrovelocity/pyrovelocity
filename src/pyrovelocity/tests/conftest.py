import os
import tempfile
import uuid
from importlib.resources import files

import mlflow
import pytest
import scanpy as sc

from pyrovelocity.analysis.analyze import top_mae_genes
from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.io.serialization import load_anndata_from_json
from pyrovelocity.tasks.data import download_dataset
from pyrovelocity.tasks.postprocess import postprocess_dataset
from pyrovelocity.tasks.preprocess import preprocess_dataset
from pyrovelocity.tasks.summarize import summarize_dataset
from pyrovelocity.tasks.train import train_dataset
from pyrovelocity.utils import generate_sample_data



def mlflow_isolation_scope(fixture_name, config):
    """Determine the appropriate scope for MLflow isolation based on execution mode."""
    if os.environ.get('PYTEST_XDIST_WORKER'):
        return "function"
    return "session"

@pytest.fixture(autouse=True, scope=mlflow_isolation_scope)
def disable_mlflow_remote_tracking(tmp_path_factory):
    """Disable MLflow remote tracking during tests to prevent network calls.

    This fixture overrides MLflow tracking configuration to use a local
    temporary directory instead of any remote server. Since tests run in
    isolated processes, we don't need to restore the original environment.
    """
    tmp_dir = tmp_path_factory.getbasetemp() / "mlflow_local"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    os.environ['MLFLOW_TRACKING_URI'] = f"file://{tmp_dir}"

    for env_var in ['MLFLOW_TRACKING_USERNAME', 'MLFLOW_TRACKING_PASSWORD', 'MLFLOW_TRACKING_TOKEN']:
        os.environ.pop(env_var, None)

    mlflow.set_tracking_uri(f"file://{tmp_dir}")

    try:
        mlflow.set_experiment("test_experiment")
    except Exception:
        try:
            mlflow.create_experiment("test_experiment")
            mlflow.set_experiment("test_experiment")
        except Exception:
            pass

    yield

# see `src/pyrovelocity/tests/fixtures/get_fixture_hashes.py` to update fixture hashes
FIXTURE_HASHES = {
    "pancreas_50_13.json": "9dcf9914f5905e248b6f6309635eddbfecd94f132a738b89381d19da3c6a2e12",
    "pancreas_raw_96_10.json": "97711767fdb13a96895450123f36fcc770c53e202154c5c1317daf12010182e5",
    "preprocessed_pancreas_50_7.json": "95c80131694f2c6449a48a56513ef79cdc56eae75204ec69abde0d81a18722ae",
    "trained_pancreas_50_7.json": "8c575d9de0430003b469b9cc9850171914a4fe1f0ae655fe0146f81af34abd04",
    "postprocessed_pancreas_50_7.json": "d50813ad23e4ae1c34f483547a7d8351fdfa94c805098caf99b8864eba8892ef",
    "larry_multilineage_50_6.json": "227a025f340e9ead0779abf8349e6c2a9774301b50e13f1c6d9f3f96001dfe73",
    "preprocessed_larry_multilineage_50_6.json": "61d3da04b5de323d3e0fd0bfd6218281c76e58f2d5271d52247f2f3218f1b1a2",
    "trained_larry_multilineage_50_6.json": "c6338e64b437e8b7a82f245729e585dc4fa7f11cd428777716e84fc6c46603f8",
    "postprocessed_larry_multilineage_50_6.json": "a8aeec31939a8d1b93577e5bf3d4f747d69cd4564bd87af54ec800903aaa25a6",
    "larry_cospar_100_6.json": "750280da645f11e6311b0ee2c3f628475f154ba9cf5235d9653cb90c50df8fde",
    "larry_mono_100_6.json": "fb9c662ee58c1a74b30700ba1e60b439871b8b1fcf2f712d3dd44add2d642ddf",
    "larry_neu_100_6.json": "47f4273f4d7ba17f914707da595b3f4fcd770176a95b62d024740e078331aa55",
}


@pytest.fixture
def adata_pancreas_50_13():
    fixture_file_path = (
        files("pyrovelocity.tests.data") / "pancreas_50_13.json"
    )
    return load_anndata_from_json(
        filename=fixture_file_path,
        expected_hash=FIXTURE_HASHES["pancreas_50_13.json"],
        sparse_layers=True,  # Convert layers to sparse matrices for cytotrace compatibility
    )


@pytest.fixture
def adata_pancreas_raw_96_10():
    fixture_file_path = (
        files("pyrovelocity.tests.data") / "pancreas_raw_96_10.json"
    )
    return load_anndata_from_json(
        filename=fixture_file_path,
        expected_hash=FIXTURE_HASHES["pancreas_raw_96_10.json"],
        sparse_layers=True,  # Convert layers to sparse matrices for scvelo compatibility
    )


@pytest.fixture
def adata_preprocessed_pancreas_50_7():
    fixture_file_path = (
        files("pyrovelocity.tests.data") / "preprocessed_pancreas_50_7.json"
    )
    return load_anndata_from_json(
        filename=fixture_file_path,
        expected_hash=FIXTURE_HASHES["preprocessed_pancreas_50_7.json"],
    )


@pytest.fixture
def adata_trained_pancreas_50_7():
    fixture_file_path = (
        files("pyrovelocity.tests.data") / "trained_pancreas_50_7.json"
    )
    return load_anndata_from_json(
        filename=fixture_file_path,
        expected_hash=FIXTURE_HASHES["trained_pancreas_50_7.json"],
    )


@pytest.fixture
def adata_postprocessed_pancreas_50_7():
    fixture_file_path = (
        files("pyrovelocity.tests.data") / "postprocessed_pancreas_50_7.json"
    )
    return load_anndata_from_json(
        filename=fixture_file_path,
        expected_hash=FIXTURE_HASHES["postprocessed_pancreas_50_7.json"],
    )


@pytest.fixture
def adata_larry_multilineage_50_6():
    fixture_file_path = (
        files("pyrovelocity.tests.data") / "larry_multilineage_50_6.json"
    )
    return load_anndata_from_json(
        filename=fixture_file_path,
        expected_hash=FIXTURE_HASHES["larry_multilineage_50_6.json"],
    )


@pytest.fixture
def adata_preprocessed_larry_multilineage_50_6():
    fixture_file_path = (
        files("pyrovelocity.tests.data")
        / "preprocessed_larry_multilineage_50_6.json"
    )
    return load_anndata_from_json(
        filename=fixture_file_path,
        expected_hash=FIXTURE_HASHES[
            "preprocessed_larry_multilineage_50_6.json"
        ],
    )


@pytest.fixture
def adata_trained_larry_multilineage_50_6():
    fixture_file_path = (
        files("pyrovelocity.tests.data")
        / "trained_larry_multilineage_50_6.json"
    )
    return load_anndata_from_json(
        filename=fixture_file_path,
        expected_hash=FIXTURE_HASHES["trained_larry_multilineage_50_6.json"],
    )


@pytest.fixture
def adata_postprocessed_larry_multilineage_50_6():
    fixture_file_path = (
        files("pyrovelocity.tests.data")
        / "postprocessed_larry_multilineage_50_6.json"
    )
    return load_anndata_from_json(
        filename=fixture_file_path,
        expected_hash=FIXTURE_HASHES[
            "postprocessed_larry_multilineage_50_6.json"
        ],
    )


@pytest.fixture
def pancreas_model2_path():
    return files("pyrovelocity.tests.data") / "models" / "pancreas_model2"


@pytest.fixture
def pancreas_model2_posterior_samples_path(pancreas_model2_path):
    return pancreas_model2_path / "posterior_samples.pkl.zst"


@pytest.fixture
def pancreas_model2_pyrovelocity_data_path(pancreas_model2_path):
    return pancreas_model2_path / "pyrovelocity.pkl.zst"


@pytest.fixture
def pancreas_model2_pyrovelocity_data(pancreas_model2_pyrovelocity_data_path):
    return CompressedPickle.load(pancreas_model2_pyrovelocity_data_path)


@pytest.fixture
def pancreas_model2_putative_marker_genes(pancreas_model2_pyrovelocity_data):
    return top_mae_genes(
        volcano_data=pancreas_model2_pyrovelocity_data["gene_ranking"],
    )


@pytest.fixture
def pancreas_model2_model_path(pancreas_model2_path):
    return pancreas_model2_path / "model"


@pytest.fixture
def pancreas_model2_metrics_path(pancreas_model2_path):
    return pancreas_model2_path / "metrics.json"


@pytest.fixture
def pancreas_model2_run_info_path(pancreas_model2_path):
    return pancreas_model2_path / "run_info.json"


@pytest.fixture
def larry_multilineage_model2_path():
    return (
        files("pyrovelocity.tests.data")
        / "models"
        / "larry_multilineage_model2"
    )


@pytest.fixture
def larry_multilineage_model2_posterior_samples_path(
    larry_multilineage_model2_path
):
    return larry_multilineage_model2_path / "posterior_samples.pkl.zst"


@pytest.fixture
def larry_multilineage_model2_pyrovelocity_data_path(
    larry_multilineage_model2_path
):
    return larry_multilineage_model2_path / "pyrovelocity.pkl.zst"


@pytest.fixture
def larry_multilineage_model2_pyrovelocity_data(
    larry_multilineage_model2_pyrovelocity_data_path
):
    return CompressedPickle.load(
        larry_multilineage_model2_pyrovelocity_data_path
    )


@pytest.fixture
def larry_multilineage_model2_putative_marker_genes(
    larry_multilineage_model2_pyrovelocity_data
):
    return top_mae_genes(
        volcano_data=larry_multilineage_model2_pyrovelocity_data[
            "gene_ranking"
        ],
    )


@pytest.fixture
def larry_multilineage_model2_model_path(larry_multilineage_model2_path):
    return larry_multilineage_model2_path / "model"


@pytest.fixture
def larry_multilineage_model2_metrics_path(larry_multilineage_model2_path):
    return larry_multilineage_model2_path / "metrics.json"


@pytest.fixture
def larry_multilineage_model2_run_info_path(larry_multilineage_model2_path):
    return larry_multilineage_model2_path / "run_info.json"


@pytest.fixture
def default_sample_data():
    """Default sample data from `pyrovelocity.utils.generate_sample_data`.

    Returns:
        AnnData: AnnData object to use for testing.
    """
    return generate_sample_data(random_seed=98)


@pytest.fixture
def default_sample_data_file(default_sample_data, tmp_path):
    """Create a .h5ad file from the default sample data."""
    file_path = tmp_path / "default_sample_data.h5ad"
    default_sample_data.write(file_path)
    return file_path


def integration_fixture_scope(fixture_name, config):
    if os.environ.get('PYTEST_XDIST_WORKER'):
        return "function"

    if config.getoption("--cached-integration-fixtures", None):
        return "session"
    return "session"


@pytest.fixture(scope=integration_fixture_scope)
def tmp_data_dir(tmp_path_factory):
    integration_tests_dir = tmp_path_factory.mktemp("integration")
    print(
        f"\nTemporary integration tests directory:\n\n",
        f"{integration_tests_dir}\n",
    )
    return integration_tests_dir


@pytest.fixture(scope=integration_fixture_scope)
def tmp_unit_reports_dir(tmp_data_dir):
    unit_reports_dir = tmp_data_dir / "unit_reports"
    unit_reports_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"\nTemporary unit reports directory:\n\n",
        f"{unit_reports_dir}\n",
    )
    return unit_reports_dir


@pytest.fixture(scope=integration_fixture_scope)
def simulated_dataset_path(tmp_data_dir):
    return download_dataset(
        data_set_name="simulated",
        data_external_path=tmp_data_dir / "data/external",
        source="simulate",
        n_obs=100,
        n_vars=200,
    )


@pytest.fixture(scope=integration_fixture_scope)
def preprocess_dataset_output(simulated_dataset_path, tmp_data_dir):
    return preprocess_dataset(
        data_set_name="simulated",
        adata=simulated_dataset_path,
        data_processed_path=tmp_data_dir / "data/processed",
        reports_processed_path=tmp_data_dir / "reports/processed",
    )


@pytest.fixture(scope=integration_fixture_scope)
def train_dataset_output(preprocess_dataset_output, tmp_data_dir):
    _, preprocessed_dataset_path, _ = preprocess_dataset_output
    return train_dataset(
        adata=preprocessed_dataset_path,
        models_path=tmp_data_dir / "models",
        max_epochs=200,
    )


@pytest.fixture(scope=integration_fixture_scope)
def data_model2_reports_path(train_dataset_output, tmp_unit_reports_dir):
    unit_reports_data_model2_path = (
        tmp_unit_reports_dir / train_dataset_output[0]
    )
    unit_reports_data_model2_path.mkdir(parents=True, exist_ok=True)
    print(
        f"\nTemporary data model reports directory:\n\n",
        f"{unit_reports_data_model2_path}\n",
    )
    return unit_reports_data_model2_path


@pytest.fixture(scope=integration_fixture_scope)
def postprocess_dataset_output(train_dataset_output):
    return postprocess_dataset(
        *train_dataset_output[:6],
        vector_field_basis="umap",
        number_posterior_samples=4,
    )


@pytest.fixture(scope=integration_fixture_scope)
def summarize_dataset_output(
    train_dataset_output,
    postprocess_dataset_output,
    tmp_data_dir,
):
    return summarize_dataset(
        *train_dataset_output[:2],
        model_path=train_dataset_output[3],
        pyrovelocity_data_path=postprocess_dataset_output[0],
        postprocessed_data_path=postprocess_dataset_output[1],
        cell_state="leiden",
        vector_field_basis="umap",
        reports_path=tmp_data_dir / "reports",
    )


@pytest.fixture(scope=integration_fixture_scope)
def train_dataset_model1_output(preprocess_dataset_output, tmp_data_dir):
    _, preprocessed_dataset_path, _ = preprocess_dataset_output
    return train_dataset(
        adata=preprocessed_dataset_path,
        model_identifier="model1",
        models_path=tmp_data_dir / "models",
        guide_type="auto_t0_constraint",
        offset=False,
        max_epochs=200,
    )


@pytest.fixture(scope=integration_fixture_scope)
def data_model1_reports_path(train_dataset_model1_output, tmp_unit_reports_dir):
    unit_reports_data_model1_path = (
        tmp_unit_reports_dir / train_dataset_model1_output[0]
    )
    unit_reports_data_model1_path.mkdir(parents=True, exist_ok=True)
    print(
        f"\nTemporary data model 1 unit reports directory:\n\n",
        f"{unit_reports_data_model1_path}\n",
    )
    return unit_reports_data_model1_path


@pytest.fixture(scope=integration_fixture_scope)
def postprocess_dataset_model1_output(train_dataset_model1_output):
    return postprocess_dataset(
        *train_dataset_model1_output[:6],
        vector_field_basis="umap",
        number_posterior_samples=4,
    )


@pytest.fixture(scope=integration_fixture_scope)
def summarize_dataset_model1_output(
    train_dataset_model1_output,
    postprocess_dataset_model1_output,
    tmp_data_dir,
):
    return summarize_dataset(
        *train_dataset_model1_output[:2],
        model_path=train_dataset_model1_output[3],
        pyrovelocity_data_path=postprocess_dataset_model1_output[0],
        postprocessed_data_path=postprocess_dataset_model1_output[1],
        cell_state="leiden",
        vector_field_basis="umap",
        reports_path=tmp_data_dir / "reports",
    )


@pytest.fixture(scope=integration_fixture_scope)
def postprocessed_model2_data(postprocess_dataset_output):
    return sc.read(postprocess_dataset_output[1])


@pytest.fixture(scope=integration_fixture_scope)
def postprocessed_model1_data(postprocess_dataset_model1_output):
    return sc.read(postprocess_dataset_model1_output[1])


@pytest.fixture(scope=integration_fixture_scope)
def posterior_samples_model2(postprocess_dataset_output):
    return CompressedPickle.load(postprocess_dataset_output[0])


@pytest.fixture(scope=integration_fixture_scope)
def putative_model2_marker_genes(posterior_samples_model2):
    return top_mae_genes(
        volcano_data=posterior_samples_model2["gene_ranking"],
    )


@pytest.fixture(scope=integration_fixture_scope)
def posterior_samples_model1(postprocess_dataset_model1_output):
    return CompressedPickle.load(postprocess_dataset_model1_output[0])


@pytest.fixture(scope=integration_fixture_scope)
def putative_model1_marker_genes(posterior_samples_model1):
    return top_mae_genes(
        volcano_data=posterior_samples_model1["gene_ranking"],
    )


# General-purpose fixtures for temporary file handling
@pytest.fixture
def temp_file_path():
    """Create a unique temporary file path for each test.

    This fixture provides a unique file path (not an actual file) for tests
    that need to write and read files. The path is automatically cleaned up
    after the test completes.

    Returns:
        str: A unique temporary file path
    """
    # Create a unique filename using uuid to avoid conflicts in parallel testing
    unique_filename = f"test_data_{uuid.uuid4().hex}"
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, unique_filename)

    yield file_path

    # Clean up after the test
    if os.path.exists(file_path):
        os.remove(file_path)


@pytest.fixture
def temp_compressed_pickle_path():
    """Create a unique temporary file path for compressed pickle files.

    This fixture provides a unique file path with the .pkl.zst extension
    for tests that need to work with CompressedPickle. The path is
    automatically cleaned up after the test completes.

    Returns:
        str: A unique temporary file path with .pkl.zst extension
    """
    # Create a unique filename using uuid to avoid conflicts in parallel testing
    unique_filename = f"test_data_{uuid.uuid4().hex}.pkl.zst"
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, unique_filename)

    yield file_path

    # Clean up after the test
    if os.path.exists(file_path):
        os.remove(file_path)


@pytest.fixture
def save_and_load_helper():
    """Helper function to save and load data with CompressedPickle.

    This fixture provides a function that handles both saving and loading
    data with CompressedPickle, making tests more concise and reliable.

    Returns:
        function: A function that saves and loads data
    """

    def _save_and_load(data, file_path, **kwargs):
        """Save data to a file and load it back.

        Args:
            data: The data to save
            file_path: The path to save the data to
            **kwargs: Additional arguments to pass to CompressedPickle.save
                      and CompressedPickle.load

        Returns:
            The loaded data
        """
        save_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["sparsify", "density_threshold"]
        }
        load_kwargs = {k: v for k, v in kwargs.items() if k in ["densify"]}

        CompressedPickle.save(file_path=file_path, obj=data, **save_kwargs)
        return CompressedPickle.load(file_path=file_path, **load_kwargs)

    return _save_and_load



@pytest.fixture(scope="session")
def larry_cospar_100_6():
    """Larry COSPAR dataset fixture with 50 cells and 5 genes.

    Optimized fixture to replace downloading 2.7GB larry_cospar.h5ad.
    Contains fate_potency_transition_map and X_emb for lineage fate correlation tests.
    Compatible with all tests requiring COSPAR data.
    """
    fixture_path = files("pyrovelocity.tests.data") / "larry_cospar_100_6.json"
    return load_anndata_from_json(
        filename=fixture_path,
        expected_hash=FIXTURE_HASHES["larry_cospar_100_6.json"],
    )


@pytest.fixture(scope="session")
def larry_mono_100_6():
    """Larry mono dataset fixture with 100 cells and 6 genes (clone-aware version).

    Optimized fixture to replace downloading 56MB larry_mono.h5ad.
    Maintains proper clone trajectory structure with 12+ good clones across time points.
    """
    fixture_path = files("pyrovelocity.tests.data") / "larry_mono_100_6.json"
    return load_anndata_from_json(
        filename=fixture_path,
        expected_hash=FIXTURE_HASHES["larry_mono_100_6.json"],
    )


@pytest.fixture(scope="session")
def larry_neu_100_6():
    """Larry neu dataset fixture with 100 cells and 6 genes (clone-aware version).

    Optimized fixture to replace downloading 50MB larry_neu.h5ad.
    Maintains proper clone trajectory structure with 14+ good clones across time points.
    """
    fixture_path = files("pyrovelocity.tests.data") / "larry_neu_100_6.json"
    return load_anndata_from_json(
        filename=fixture_path,
        expected_hash=FIXTURE_HASHES["larry_neu_100_6.json"],
    )
