-- Multi-Dataset DuckLake Example
-- Demonstrates how variables make it easy to work with multiple datasets

-- =============================================================================
-- SETUP
-- =============================================================================

-- Install and load extensions
INSTALL ducklake; INSTALL sqlite; INSTALL httpfs;
LOAD ducklake; LOAD sqlite; LOAD httpfs;

-- Note: Using macros instead of variables for duckdb-wasm compatibility
-- Create macros for dataset paths (DRY - includes .pqdata/ and .parquet)
CREATE OR REPLACE MACRO pancreas_path(file) AS 'postprocessed_pancreas_50_7.pqdata/' || file || '.parquet';
CREATE OR REPLACE MACRO bifurcation_path(file) AS 'bifurcation_14.pqdata/' || file || '.parquet';

-- legacy settings-based s3 configuration
-- SET s3_region = 'us-central1';
-- SET s3_endpoint = 'https://storage.googleapis.com';

-- secret-based s3 configuration
CREATE OR REPLACE SECRET gcs_pv (
    TYPE gcs,
    KEY_ID getenv('GCS_HMAC_ACCESS_ID'),
    SECRET getenv('GCS_HMAC_SECRET_ACCESS_KEY')
);


-- Create DuckLake with local DATA_PATH (DuckLake needs local filesystem)
ATTACH OR REPLACE 'ducklake:sqlite:multi_dataset_gcs.sqlite' AS multi_lake (DATA_PATH 'gs://pyrovelocity/lake/fixtures/');
USE multi_lake;

-- =============================================================================
-- DATASET 1: PANCREAS
-- =============================================================================

SELECT 'Ingesting pancreas dataset...' as step;

-- Create schema for pancreas
CREATE OR REPLACE SCHEMA pancreas;

-- Switch to pancreas schema for table creation
USE pancreas;

-- Ingest core components
CREATE TABLE obs AS
SELECT * FROM read_parquet(pancreas_path('obs'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'obs',
    pancreas_path('obs'));

CREATE TABLE var AS
SELECT * FROM read_parquet(pancreas_path('var'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'var',
    pancreas_path('var'));

CREATE TABLE X AS
SELECT * FROM read_parquet(pancreas_path('X'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'X',
    pancreas_path('X'));

-- Embeddings (obsm)
CREATE TABLE X_pca AS
SELECT * FROM read_parquet(pancreas_path('obsm/X_pca'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'X_pca',
    pancreas_path('obsm/X_pca'));

CREATE TABLE X_umap AS
SELECT * FROM read_parquet(pancreas_path('obsm/X_umap'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'X_umap',
    pancreas_path('obsm/X_umap'));

CREATE TABLE velocity_pyro_pca AS
SELECT * FROM read_parquet(pancreas_path('obsm/velocity_pyro_pca'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'velocity_pyro_pca',
    pancreas_path('obsm/velocity_pyro_pca'));

CREATE TABLE velocity_pyro_umap AS
SELECT * FROM read_parquet(pancreas_path('obsm/velocity_pyro_umap'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'velocity_pyro_umap',
    pancreas_path('obsm/velocity_pyro_umap'));

CREATE TABLE velocity_umap AS
SELECT * FROM read_parquet(pancreas_path('obsm/velocity_umap'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'velocity_umap',
    pancreas_path('obsm/velocity_umap'));

-- Expression layers
CREATE TABLE spliced AS
SELECT * FROM read_parquet(pancreas_path('layers/spliced'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'spliced',
    pancreas_path('layers/spliced'));

CREATE TABLE unspliced AS
SELECT * FROM read_parquet(pancreas_path('layers/unspliced'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'unspliced',
    pancreas_path('layers/unspliced'));

CREATE TABLE raw_spliced AS
SELECT * FROM read_parquet(pancreas_path('layers/raw_spliced'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'raw_spliced',
    pancreas_path('layers/raw_spliced'));

CREATE TABLE raw_unspliced AS
SELECT * FROM read_parquet(pancreas_path('layers/raw_unspliced'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'raw_unspliced',
    pancreas_path('layers/raw_unspliced'));

-- Velocity layers
CREATE TABLE velocity AS
SELECT * FROM read_parquet(pancreas_path('layers/velocity'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'velocity',
    pancreas_path('layers/velocity'));

CREATE TABLE velocity_pyro AS
SELECT * FROM read_parquet(pancreas_path('layers/velocity_pyro'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'velocity_pyro',
    pancreas_path('layers/velocity_pyro'));

CREATE TABLE velocity_u AS
SELECT * FROM read_parquet(pancreas_path('layers/velocity_u'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'velocity_u',
    pancreas_path('layers/velocity_u'));

-- Model-specific layers
CREATE TABLE Ms AS
SELECT * FROM read_parquet(pancreas_path('layers/Ms'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'Ms',
    pancreas_path('layers/Ms'));

CREATE TABLE Mu AS
SELECT * FROM read_parquet(pancreas_path('layers/Mu'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'Mu',
    pancreas_path('layers/Mu'));

CREATE TABLE fit_t AS
SELECT * FROM read_parquet(pancreas_path('layers/fit_t'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'fit_t',
    pancreas_path('layers/fit_t'));

CREATE TABLE fit_tau AS
SELECT * FROM read_parquet(pancreas_path('layers/fit_tau'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'fit_tau',
    pancreas_path('layers/fit_tau'));

CREATE TABLE fit_tau_ AS
SELECT * FROM read_parquet(pancreas_path('layers/fit_tau_'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'fit_tau_',
    pancreas_path('layers/fit_tau_'));

CREATE TABLE spliced_pyro AS
SELECT * FROM read_parquet(pancreas_path('layers/spliced_pyro'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'spliced_pyro',
    pancreas_path('layers/spliced_pyro'));

-- =============================================================================
-- DATASET 2: BIFURCATION
-- =============================================================================

SELECT 'Ingesting bifurcation dataset...' as step;

-- Create schema for bifurcation
CREATE OR REPLACE SCHEMA bifurcation;

-- Switch to bifurcation schema for table creation
USE bifurcation;

-- Ingest core components (same pattern, different dataset)
CREATE TABLE obs AS
SELECT * FROM read_parquet(bifurcation_path('obs'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'obs',
    bifurcation_path('obs'));

CREATE TABLE var AS
SELECT * FROM read_parquet(bifurcation_path('var'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'var',
    bifurcation_path('var'));

CREATE TABLE X AS
SELECT * FROM read_parquet(bifurcation_path('X'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'X',
    bifurcation_path('X'));

-- Embeddings (obsm)
CREATE TABLE X_pca AS
SELECT * FROM read_parquet(bifurcation_path('obsm/X_pca'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'X_pca',
    bifurcation_path('obsm/X_pca'));

CREATE TABLE true_sc_network AS
SELECT * FROM read_parquet(bifurcation_path('obsm/true_sc_network'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'true_sc_network',
    bifurcation_path('obsm/true_sc_network'));

-- Expression layers
CREATE TABLE spliced AS
SELECT * FROM read_parquet(bifurcation_path('layers/spliced'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'spliced',
    bifurcation_path('layers/spliced'));

CREATE TABLE unspliced AS
SELECT * FROM read_parquet(bifurcation_path('layers/unspliced'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'unspliced',
    bifurcation_path('layers/unspliced'));

CREATE TABLE spliced_raw AS
SELECT * FROM read_parquet(bifurcation_path('layers/spliced_raw'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'spliced_raw',
    bifurcation_path('layers/spliced_raw'));

CREATE TABLE unspliced_raw AS
SELECT * FROM read_parquet(bifurcation_path('layers/unspliced_raw'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'unspliced_raw',
    bifurcation_path('layers/unspliced_raw'));

-- Velocity layers
CREATE TABLE velocity AS
SELECT * FROM read_parquet(bifurcation_path('layers/velocity'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'velocity',
    bifurcation_path('layers/velocity'));

CREATE TABLE true_velocity AS
SELECT * FROM read_parquet(bifurcation_path('layers/true_velocity'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'true_velocity',
    bifurcation_path('layers/true_velocity'));

-- Model-specific layers
CREATE TABLE Ms AS
SELECT * FROM read_parquet(bifurcation_path('layers/Ms'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'Ms',
    bifurcation_path('layers/Ms'));

CREATE TABLE Mu AS
SELECT * FROM read_parquet(bifurcation_path('layers/Mu'))
LIMIT 0;

CALL ducklake_add_data_files('multi_lake', 'Mu',
    bifurcation_path('layers/Mu'));

-- Switch back to main database for verification queries
USE multi_lake;

-- =============================================================================
-- VERIFICATION
-- =============================================================================

SELECT 'Verifying multi-dataset catalog...' as step;

-- Show all schemas
SELECT schema_name 
FROM information_schema.schemata 
WHERE schema_name IN ('pancreas', 'bifurcation');

-- Show all tables
SELECT table_schema, table_name 
FROM information_schema.tables 
WHERE table_schema IN ('pancreas', 'bifurcation')
ORDER BY table_schema, table_name;

-- Compare dataset sizes
SELECT 'Dataset comparison:' as info;
SELECT 
    'pancreas' as dataset,
    (SELECT COUNT(*) FROM pancreas.obs) as cells,
    (SELECT COUNT(*) FROM pancreas.var) as genes
UNION ALL
SELECT 
    'bifurcation' as dataset,
    (SELECT COUNT(*) FROM bifurcation.obs) as cells,
    (SELECT COUNT(*) FROM bifurcation.var) as genes;

-- =============================================================================
-- ADDING NEW DATASETS
-- =============================================================================

-- To add a new dataset, just:
-- 1. CREATE OR REPLACE MACRO new_dataset_path(file) AS 'new_dataset_name.pqdata/' || file || '.parquet';
-- 2. CREATE SCHEMA new_dataset_name;
-- 3. Use new_dataset_path('file') pattern

USE memory; DETACH multi_lake;
ATTACH 'ducklake:sqlite:multi_dataset_gcs.sqlite' AS multi_lake (READ_ONLY);
