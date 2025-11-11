# src/plot.py
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from google.cloud import storage

def save_and_upload_fig(fig, local_path, bucket_name, gcs_path):
    """Save the figure locally and upload it to GCS."""
    fig.savefig(local_path, bbox_inches='tight')
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"✅ Uploaded figure to gs://{bucket_name}/{gcs_path}")
    plt.close(fig)

def plot_residuals(preds_df, label_col, model_name, bucket_name):
    """Plot residuals and log summary statistics."""
    pred_pd = preds_df.select(label_col, "prediction").toPandas()
    residuals = pred_pd[label_col] - pred_pd["prediction"]
    summary = residuals.describe()
    print("Residual summary:\n" + summary.to_string())
    print("Residual 95/99/99.9 percentiles:\n" + residuals.quantile([0.95, 0.99, 0.999]).to_string())
    fig, ax = plt.subplots()
    ax.plot(residuals.values, label=model_name.upper())
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Plot")
    ax.legend()
    local_path = f"{model_name}_residuals.png"
    gcs_path = f"figures/{local_path}"
    save_and_upload_fig(fig, local_path, bucket_name, gcs_path)

def plot_pred_vs_actual(preds_df, label_col, model_name, bucket_name):
    """Scatter plot comparing predictions to ground truth."""
    pred_pd = preds_df.select(label_col, "prediction").toPandas()

    # Scatter plot
    fig, ax = plt.subplots()
    ax.scatter(pred_pd[label_col], pred_pd["prediction"], alpha=0.5)
    ax.plot([pred_pd[label_col].min(), pred_pd[label_col].max()],
            [pred_pd[label_col].min(), pred_pd[label_col].max()],
            color='red', linestyle='--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{model_name.upper()} Predicted vs Actual")
    local_path = f"{model_name}_scatter.png"
    gcs_path = f"figures/{local_path}"
    save_and_upload_fig(fig, local_path, bucket_name, gcs_path)

def plot_time_series(pred_df, label_col, time_col, model_name, bucket_name):
    """Plot actual versus predicted values across time."""

    if time_col not in pred_df.columns:
        return
    pred_pd = pred_df.select(time_col, label_col, "prediction").toPandas()
    fig, ax = plt.subplots()
    ax.plot(pred_pd[time_col], pred_pd[label_col], label="Actual")
    ax.plot(pred_pd[time_col], pred_pd["prediction"], label="Predicted")
    ax.set_xlabel("Time")
    ax.set_ylabel(label_col)
    ax.set_title(f"{model_name.upper()} Actual vs Predicted over Time")
    ax.legend()
    local_path = f"{model_name}_timeseries.png"
    gcs_path = f"figures/{local_path}"
    save_and_upload_fig(fig, local_path, bucket_name, gcs_path)

def plot_feature_importances(model_stage, numeric_features, model_name, bucket_name):
    """Visualize feature importances for tree-based models."""
    if not hasattr(model_stage, "featureImportances"):
        return
    # Convert to numpy for sorting
    feature_importances = np.array(model_stage.featureImportances)
    fig, ax = plt.subplots(figsize=(10,6))
    indices = np.argsort(feature_importances)[::-1]
    ax.bar(range(len(numeric_features)), feature_importances[indices])
    ax.set_xticks(range(len(numeric_features)))
    ax.set_xticklabels(
        [numeric_features[i] for i in indices],
        rotation=45,
        ha="right"
    )
    ax.set_ylabel("Feature Importance")
    ax.set_title(f"{model_name.upper()} Feature Importances")
    fig.tight_layout()
    local_path = f"{model_name}_feature_importance.png"
    gcs_path = f"figures/{local_path}"
    save_and_upload_fig(fig, local_path, bucket_name, gcs_path)

def plot_loss_curve(model_stage, model_name, bucket_name):
    """Plot the training loss curve for GBT or Elastic Net models."""
    if hasattr(model_stage, "summary") and model_stage.summary is not None:
        try:
            loss_history = model_stage.summary.loss
        except AttributeError:
            loss_history = model_stage.summary.objectiveHistory
        fig, ax = plt.subplots()
        ax.plot(range(len(loss_history)), loss_history, marker='o')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Training Loss")
        ax.set_title(f"{model_name.upper()} Training Loss Curve")
        ax.grid(True)
        local_path = f"{model_name}_loss_curve.png"
        gcs_path = f"figures/{local_path}"
        save_and_upload_fig(fig, local_path, bucket_name, gcs_path)
