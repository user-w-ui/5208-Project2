# Weather Forecasting on Dataproc

PySpark pipeline for cleaning NOAA ISD observations, engineering features, training regressors, and exporting diagnostics to Google Cloud Storage (GCS). Jobs can run locally or on a Dataproc cluster.

## Directory Map

| Path | Purpose |
|------|---------|
| `src/` | Library code for reusable components. `time_cv.py` builds chronological folds, `model_selection.py` runs grid search, `plot.py` centralises plotting and GCS upload helpers. |
| `scripts/` | Stand-alone scripts for data ingestion, feature engineering, and Dataproc submission (see next section for details). |
| `test/` | PySpark entrypoints executed directly or through Dataproc submissions (`main.py`, `plot_result.py`, `visualize.py`, experimental `main_gpu.py`). |
| `notebooks/` | Documentation notebooks demonstrating cleaning and training workflows. |
| `figures/` *(created locally)* | Downloaded charts when copying from GCS. |
| `models/` *(created at runtime)* | Saved Spark pipelines; mirrored to `gs://<bucket>/models/<model>_model/`. |

## Script Reference (`scripts/`)

| Script | Summary |
|--------|---------|
| `extract.py`, `test/extract.py` | Stream `.tar.gz` archives from GCS, extract CSV members on the fly, and push them back to the destination bucket without persisting to disk. |
| `data_processing.py` | Production cleaning workflow: station filtering, anomaly removal, interpolation, periodic features, and chronological train/test split. |
| `data_processing_part1.py` | Exploratory processing stage that produces histogram plots and uploads them to GCS. |
| `data_processing_new.py`, `test/data_processing_new.py` | Iterative cleaning scripts used for experiments and debugging smaller subsets. |
| `time_cv.py` | Chronological splitter utilities reused by training scripts. |
| `features.py` | Detects numeric feature columns while excluding label/timestamp fields. |
| `submit_main.sh` | Packages `src/` into `src.zip`, submits `test/main.py` (supports `rf`, `gbrt`, `elastic`), copies logs to `gs://<bucket>/logs/`, uploads `models/<model>_model/`, and deletes the cluster at the end (comment out the delete command to keep the cluster). |
| `submit_plotresult.sh` | Submits `test/plot_result.py` for a fixed best-parameter run that regenerates metrics and figures. Logs go to `gs://<bucket>/logs/`; enable model upload by ensuring the job writes `models/<model>_model/`. |
| `submit_visualize.sh` | Runs `test/visualize.py` to sweep single hyperparameters and upload comparison plots. |

All submission scripts default to cluster `my-cluster`, region `asia-southeast1`, and bucket `spark-result-lyx`. Adjust the variables at the top of each script if your environment differs.

## Entry Points (`test/`)

| File | Role |
|------|------|
| `main.py` | Main training driver. Performs sub-sampling, prefix cross-validation, prints RMSE/MAE/R², logs Elastic Net coefficients, saves the pipeline locally, and mirrors artifacts to GCS. |
| `plot_result.py` | Executes a single model configuration, reports Train/Test RMSE, MAE, R², MAPE, prints residual summaries, and generates plots (residuals, scatter, time series, loss curves). Used by `submit_plotresult.sh`. |
| `visualize.py` | One-parameter sweep for GBRT and Elastic Net; produces charts that land in `gs://<bucket>/figures/`. |
| `main_gpu.py` | Minimal driver for GPU or reduced-feature experiments. |

## Plotting Outputs

`src/plot.py` saves each figure locally (e.g., `elastic_residuals.png`) before uploading to `gs://<bucket>/figures/`. Residual statistics (describe + 95/99/99.9 percentiles) are printed to the job log to highlight outliers. Feature-importance charts rotate x-labels 45° for readability.

## Notebooks

- `notebooks/data_processing_demo.ipynb` – step-by-step cleaning walkthrough mirroring the production scripts.
- `notebooks/train_framework.ipynb` – explains prefix CV/grid search helpers and compares model metrics.
- `notebooks/plot_elastic_param.ipynb` – coefficient visualisations saved into `figures/`.

Use notebooks for analysis; scripted runs should use the corresponding files in `scripts/` and `test/`.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install pyspark google-cloud-storage pandas matplotlib seaborn
gcloud auth login
gcloud config set project <your-project-id>
```

Ensure the Dataproc service account has `storage.objects.get` and `storage.objects.create` on the target bucket.

## Running Locally

```bash
python test/main.py \
  --model elastic \
  --train-path gs://spark-result-lyx/train_withds \
  --test-path gs://spark-result-lyx/test_withds \
  --sample-fraction 0.001 \
  --num-folds 4 \
  --bucket spark-result-lyx
```

Artifacts appear under `models/<model>_model/` and `gs://<bucket>/figures/`. Increase `--sample-fraction` cautiously if memory allows.

## Running on Dataproc

1. (Optional) create a cluster:
   ```bash
   gcloud dataproc clusters create my-cluster \
     --region=asia-southeast1 \
     --num-workers=2 \
     --worker-machine-type=n2-standard-4 \
     --master-machine-type=n2-standard-4 \
     --image-version=2.2-debian12
   ```
2. Submit the training job:
   ```bash
   chmod +x scripts/submit_main.sh
   ./scripts/submit_main.sh elastic
   ```
3. Regenerate plots/metrics only:
   ```bash
   chmod +x scripts/submit_plotresult.sh
   ./scripts/submit_plotresult.sh elastic
   ```

Logs are written to `/tmp/job_<model>.log` and copied to `gs://<bucket>/logs/main_<model>.log`. Plots land in `gs://<bucket>/figures/`; trained pipelines save to both the local `models/` directory and `gs://<bucket>/models/<model>_model/`.

## Customising Hyperparameters

- Update the `param_grid` dictionaries in `test/main.py` for grid search.
- Adjust the fixed parameter dictionary in `test/plot_result.py` when you want `submit_plotresult.sh` to test new settings.
- `test/visualize.py` contains sweep ranges for quick sensitivity analysis.

## Troubleshooting

- `403 Forbidden` when reading parquet → grant the Dataproc service account `storage.objects.get` on the bucket or update the bucket name in the submission script.  
- `CommandException: No URLs matched: models/<model>_model` → ensure the job saved a pipeline locally (see the save block near the end of `test/main.py` or re-enable it in `test/plot_result.py`).  
- Residual plot spans ±20 despite low RMSE → inspect the printed percentiles to confirm only a few outliers drive the spikes.

## Cleanup

```bash
rm -rf src.zip models/
gsutil rm -r gs://spark-result-lyx/logs/
gcloud dataproc clusters delete my-cluster --region=asia-southeast1
```

Keep bucket names, regions, and cluster settings in sync with your project. Update this README as new scripts or workflows are added.
