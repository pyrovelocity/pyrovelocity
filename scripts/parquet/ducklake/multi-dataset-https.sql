-- Multi-Dataset DuckLake Example
-- Approximately equivalent to attaching a remote DuckDB database:
--
-- ATTACH 'https://huggingface.co/datasets/pyrovelocity/fixtures/resolve/main/example.db' as example (READ_ONLY);
--
-- but using parquet files instead of a pre-built database.
-- Executable with: ./duckdb -init multi-dataset-https.sql

-- =============================================================================
-- SETUP
-- =============================================================================

-- Install and load extensions
INSTALL ducklake; INSTALL sqlite; INSTALL httpfs;
LOAD ducklake; LOAD sqlite; LOAD httpfs;

-- Note: Using macros instead of variables for duckdb-wasm compatibility
-- Create macros for dataset paths (DRY - includes .pqdata/ and .parquet)
CREATE OR REPLACE MACRO pancreas_path(file) AS 'https://huggingface.co/datasets/pyrovelocity/fixtures/resolve/main/postprocessed_pancreas_50_7.pqdata/' || file || '.parquet';
CREATE OR REPLACE MACRO bifurcation_path(file) AS 'https://huggingface.co/datasets/pyrovelocity/fixtures/resolve/main/bifurcation_14.pqdata/' || file || '.parquet';

-- Create DuckLake with local DATA_PATH (DuckLake needs local filesystem)
ATTACH OR REPLACE 'ducklake:sqlite:multi_dataset_https.sqlite' AS multi_lake (DATA_PATH 'multi_dataset_https');
USE multi_lake;

-- =============================================================================
-- DATASET 1: PANCREAS
-- =============================================================================

SELECT 'Ingesting pancreas dataset...' as step;

-- Create schema for pancreas
CREATE OR REPLACE SCHEMA multi_lake.pancreas;

-- Ingest core components (fully qualified for duckdb-wasm compatibility)
CREATE TABLE multi_lake.pancreas.obs AS
SELECT * FROM read_parquet(pancreas_path('obs'));

CREATE TABLE multi_lake.pancreas.var AS
SELECT * FROM read_parquet(pancreas_path('var'));

CREATE TABLE multi_lake.pancreas.X AS
SELECT * FROM read_parquet(pancreas_path('X'));

-- Embeddings (obsm)
CREATE TABLE multi_lake.pancreas.X_pca AS
SELECT * FROM read_parquet(pancreas_path('obsm/X_pca'));

CREATE TABLE multi_lake.pancreas.X_umap AS
SELECT * FROM read_parquet(pancreas_path('obsm/X_umap'));

CREATE TABLE multi_lake.pancreas.velocity_pyro_pca AS
SELECT * FROM read_parquet(pancreas_path('obsm/velocity_pyro_pca'));

CREATE TABLE multi_lake.pancreas.velocity_pyro_umap AS
SELECT * FROM read_parquet(pancreas_path('obsm/velocity_pyro_umap'));

CREATE TABLE multi_lake.pancreas.velocity_umap AS
SELECT * FROM read_parquet(pancreas_path('obsm/velocity_umap'));

-- Expression layers
CREATE TABLE multi_lake.pancreas.spliced AS
SELECT * FROM read_parquet(pancreas_path('layers/spliced'));

CREATE TABLE multi_lake.pancreas.unspliced AS
SELECT * FROM read_parquet(pancreas_path('layers/unspliced'));

CREATE TABLE multi_lake.pancreas.raw_spliced AS
SELECT * FROM read_parquet(pancreas_path('layers/raw_spliced'));

CREATE TABLE multi_lake.pancreas.raw_unspliced AS
SELECT * FROM read_parquet(pancreas_path('layers/raw_unspliced'));

-- Velocity layers
CREATE TABLE multi_lake.pancreas.velocity AS
SELECT * FROM read_parquet(pancreas_path('layers/velocity'));

CREATE TABLE multi_lake.pancreas.velocity_pyro AS
SELECT * FROM read_parquet(pancreas_path('layers/velocity_pyro'));

CREATE TABLE multi_lake.pancreas.velocity_u AS
SELECT * FROM read_parquet(pancreas_path('layers/velocity_u'));

-- Model-specific layers
CREATE TABLE multi_lake.pancreas.Ms AS
SELECT * FROM read_parquet(pancreas_path('layers/Ms'));

CREATE TABLE multi_lake.pancreas.Mu AS
SELECT * FROM read_parquet(pancreas_path('layers/Mu'));

CREATE TABLE multi_lake.pancreas.fit_t AS
SELECT * FROM read_parquet(pancreas_path('layers/fit_t'));

CREATE TABLE multi_lake.pancreas.fit_tau AS
SELECT * FROM read_parquet(pancreas_path('layers/fit_tau'));

CREATE TABLE multi_lake.pancreas.fit_tau_ AS
SELECT * FROM read_parquet(pancreas_path('layers/fit_tau_'));

CREATE TABLE multi_lake.pancreas.spliced_pyro AS
SELECT * FROM read_parquet(pancreas_path('layers/spliced_pyro'));

-- =============================================================================
-- DATASET 2: BIFURCATION
-- =============================================================================

SELECT 'Ingesting bifurcation dataset...' as step;

-- Create schema for bifurcation
CREATE OR REPLACE SCHEMA multi_lake.bifurcation;

-- Ingest core components (fully qualified for duckdb-wasm compatibility)
CREATE TABLE multi_lake.bifurcation.obs AS
SELECT * FROM read_parquet(bifurcation_path('obs'));

CREATE TABLE multi_lake.bifurcation.var AS
SELECT * FROM read_parquet(bifurcation_path('var'));

CREATE TABLE multi_lake.bifurcation.X AS
SELECT * FROM read_parquet(bifurcation_path('X'));

-- Embeddings (obsm)
CREATE TABLE multi_lake.bifurcation.X_pca AS
SELECT * FROM read_parquet(bifurcation_path('obsm/X_pca'));

CREATE TABLE multi_lake.bifurcation.true_sc_network AS
SELECT * FROM read_parquet(bifurcation_path('obsm/true_sc_network'));

-- Expression layers
CREATE TABLE multi_lake.bifurcation.spliced AS
SELECT * FROM read_parquet(bifurcation_path('layers/spliced'));

CREATE TABLE multi_lake.bifurcation.unspliced AS
SELECT * FROM read_parquet(bifurcation_path('layers/unspliced'));

CREATE TABLE multi_lake.bifurcation.spliced_raw AS
SELECT * FROM read_parquet(bifurcation_path('layers/spliced_raw'));

CREATE TABLE multi_lake.bifurcation.unspliced_raw AS
SELECT * FROM read_parquet(bifurcation_path('layers/unspliced_raw'));

-- Velocity layers
CREATE TABLE multi_lake.bifurcation.velocity AS
SELECT * FROM read_parquet(bifurcation_path('layers/velocity'));

CREATE TABLE multi_lake.bifurcation.true_velocity AS
SELECT * FROM read_parquet(bifurcation_path('layers/true_velocity'));

-- Model-specific layers
CREATE TABLE multi_lake.bifurcation.Ms AS
SELECT * FROM read_parquet(bifurcation_path('layers/Ms'));

CREATE TABLE multi_lake.bifurcation.Mu AS
SELECT * FROM read_parquet(bifurcation_path('layers/Mu'));

-- =============================================================================
-- VERIFICATION
-- =============================================================================

SELECT 'Verifying multi-dataset catalog...' as step;

-- Show all schemas (fully qualified)
SELECT schema_name
FROM information_schema.schemata
WHERE schema_name IN ('pancreas', 'bifurcation');

-- Show all tables (fully qualified)
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_schema IN ('pancreas', 'bifurcation')
ORDER BY table_schema, table_name;

-- Compare dataset sizes (fully qualified)
SELECT 'Dataset comparison:' as info;
SELECT
    'pancreas' as dataset,
    (SELECT COUNT(*) FROM multi_lake.pancreas.obs) as cells,
    (SELECT COUNT(*) FROM multi_lake.pancreas.var) as genes
UNION ALL
SELECT
    'bifurcation' as dataset,
    (SELECT COUNT(*) FROM multi_lake.bifurcation.obs) as cells,
    (SELECT COUNT(*) FROM multi_lake.bifurcation.var) as genes;

-- =============================================================================
-- DETACH AND REATTACH
-- =============================================================================

-- To add a new dataset, just:
-- 1. CREATE OR REPLACE MACRO new_dataset_path(file) AS 'https://huggingface.co/datasets/pyrovelocity/fixtures/resolve/main/new_dataset_name.pqdata/' || file || '.parquet';
-- 2. CREATE SCHEMA new_dataset_name;
-- 3. Use new_dataset_path('file') pattern

DETACH multi_lake;
ATTACH 'ducklake:sqlite:multi_dataset_https.sqlite' AS multi_lake (READ_ONLY);
