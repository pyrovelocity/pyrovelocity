-- Multi-Dataset Standard DuckDB Example
-- Uses schema-based organization with local parquet files
-- Executable with: ./duckdb multi_dataset_analysis.db -init multi-dataset-standard-duckdb.sql

-- =============================================================================
-- SETUP
-- =============================================================================

SELECT 'Setting up multi-dataset DuckDB database...' as step;

-- Note: Using macros instead of variables for duckdb-wasm compatibility
-- Create macros for dataset paths (DRY - includes .pqdata/ and .parquet)
CREATE OR REPLACE MACRO pancreas_path(file) AS 'postprocessed_pancreas_50_7.pqdata/' || file || '.parquet';
CREATE OR REPLACE MACRO bifurcation_path(file) AS 'bifurcation_14.pqdata/' || file || '.parquet';

-- Create schemas for each dataset
CREATE SCHEMA pancreas;
CREATE SCHEMA bifurcation;

-- =============================================================================
-- DATASET 1: PANCREAS (postprocessed_pancreas_50_7.pqdata)
-- =============================================================================

SELECT 'Loading pancreas dataset...' as step;

-- Core tables
CREATE TABLE pancreas.obs AS
SELECT * FROM read_parquet(pancreas_path('obs'));

CREATE TABLE pancreas.var AS
SELECT * FROM read_parquet(pancreas_path('var'));

CREATE TABLE pancreas.X AS
SELECT * FROM read_parquet(pancreas_path('X'));

-- Embeddings (obsm)
CREATE TABLE pancreas.X_pca AS
SELECT * FROM read_parquet(pancreas_path('obsm/X_pca'));

CREATE TABLE pancreas.X_umap AS
SELECT * FROM read_parquet(pancreas_path('obsm/X_umap'));

CREATE TABLE pancreas.velocity_pyro_pca AS
SELECT * FROM read_parquet(pancreas_path('obsm/velocity_pyro_pca'));

CREATE TABLE pancreas.velocity_pyro_umap AS
SELECT * FROM read_parquet(pancreas_path('obsm/velocity_pyro_umap'));

CREATE TABLE pancreas.velocity_umap AS
SELECT * FROM read_parquet(pancreas_path('obsm/velocity_umap'));

-- Expression layers
CREATE TABLE pancreas.spliced AS
SELECT * FROM read_parquet(pancreas_path('layers/spliced'));

CREATE TABLE pancreas.unspliced AS
SELECT * FROM read_parquet(pancreas_path('layers/unspliced'));

CREATE TABLE pancreas.raw_spliced AS
SELECT * FROM read_parquet(pancreas_path('layers/raw_spliced'));

CREATE TABLE pancreas.raw_unspliced AS
SELECT * FROM read_parquet(pancreas_path('layers/raw_unspliced'));

-- Velocity layers
CREATE TABLE pancreas.velocity AS
SELECT * FROM read_parquet(pancreas_path('layers/velocity'));

CREATE TABLE pancreas.velocity_pyro AS
SELECT * FROM read_parquet(pancreas_path('layers/velocity_pyro'));

CREATE TABLE pancreas.velocity_u AS
SELECT * FROM read_parquet(pancreas_path('layers/velocity_u'));

-- Model-specific layers
CREATE TABLE pancreas.Ms AS
SELECT * FROM read_parquet(pancreas_path('layers/Ms'));

CREATE TABLE pancreas.Mu AS
SELECT * FROM read_parquet(pancreas_path('layers/Mu'));

CREATE TABLE pancreas.fit_t AS
SELECT * FROM read_parquet(pancreas_path('layers/fit_t'));

CREATE TABLE pancreas.fit_tau AS
SELECT * FROM read_parquet(pancreas_path('layers/fit_tau'));

CREATE TABLE pancreas.fit_tau_ AS
SELECT * FROM read_parquet(pancreas_path('layers/fit_tau_'));

CREATE TABLE pancreas.spliced_pyro AS
SELECT * FROM read_parquet(pancreas_path('layers/spliced_pyro'));

-- =============================================================================
-- DATASET 2: BIFURCATION (bifurcation_14.pqdata)
-- =============================================================================

SELECT 'Loading bifurcation dataset...' as step;

-- Core tables
CREATE TABLE bifurcation.obs AS
SELECT * FROM read_parquet(bifurcation_path('obs'));

CREATE TABLE bifurcation.var AS
SELECT * FROM read_parquet(bifurcation_path('var'));

CREATE TABLE bifurcation.X AS
SELECT * FROM read_parquet(bifurcation_path('X'));

-- Embeddings (obsm)
CREATE TABLE bifurcation.X_pca AS
SELECT * FROM read_parquet(bifurcation_path('obsm/X_pca'));

CREATE TABLE bifurcation.true_sc_network AS
SELECT * FROM read_parquet(bifurcation_path('obsm/true_sc_network'));

-- Expression layers
CREATE TABLE bifurcation.spliced AS
SELECT * FROM read_parquet(bifurcation_path('layers/spliced'));

CREATE TABLE bifurcation.unspliced AS
SELECT * FROM read_parquet(bifurcation_path('layers/unspliced'));

CREATE TABLE bifurcation.spliced_raw AS
SELECT * FROM read_parquet(bifurcation_path('layers/spliced_raw'));

CREATE TABLE bifurcation.unspliced_raw AS
SELECT * FROM read_parquet(bifurcation_path('layers/unspliced_raw'));

-- Velocity layers
CREATE TABLE bifurcation.velocity AS
SELECT * FROM read_parquet(bifurcation_path('layers/velocity'));

CREATE TABLE bifurcation.true_velocity AS
SELECT * FROM read_parquet(bifurcation_path('layers/true_velocity'));

-- Model-specific layers
CREATE TABLE bifurcation.Ms AS
SELECT * FROM read_parquet(bifurcation_path('layers/Ms'));

CREATE TABLE bifurcation.Mu AS
SELECT * FROM read_parquet(bifurcation_path('layers/Mu'));
