# src/plot.py
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from google.cloud import storage

def save_and_upload_fig(fig, local_path, bucket_name, gcs_path):
    """保存图并上传到GCS"""
    fig.savefig(local_path, bbox_inches='tight')
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"✅ Uploaded figure to gs://{bucket_name}/{gcs_path}")
    plt.close(fig)

def plot_residuals(preds_df, label_col, model_name, bucket_name):
    """残差图"""
    pred_pd = preds_df.select(label_col, "prediction").toPandas()
    residuals = pred_pd[label_col] - pred_pd["prediction"]
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
    """散点图与时间序列图"""
    pred_pd = preds_df.select(label_col, "prediction").toPandas()

    # 散点图
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

def plot_time_series(preds_df, test_df, label_col, time_col, model_name, bucket_name):
    """时间序列折线图"""
    if time_col not in test_df.columns:
        return
    time_pd = test_df.select(time_col).toPandas()
    pred_pd = preds_df.select(label_col, "prediction").toPandas()
    fig, ax = plt.subplots()
    ax.plot(time_pd[time_col], pred_pd[label_col], label="Actual")
    ax.plot(time_pd[time_col], pred_pd["prediction"], label="Predicted")
    ax.set_xlabel("Time")
    ax.set_ylabel(label_col)
    ax.set_title(f"{model_name.upper()} Actual vs Predicted over Time")
    ax.legend()
    local_path = f"{model_name}_timeseries.png"
    gcs_path = f"figures/{local_path}"
    save_and_upload_fig(fig, local_path, bucket_name, gcs_path)

def plot_feature_importances(model_stage, numeric_features, model_name, bucket_name):
    """树模型的特征重要性"""
    if not hasattr(model_stage, "featureImportances"):
        return
    # 转 numpy
    feature_importances = np.array(model_stage.featureImportances)
    fig, ax = plt.subplots(figsize=(10,6))
    indices = np.argsort(feature_importances)[::-1]
    ax.bar(range(len(numeric_features)), feature_importances[indices])
    ax.set_xticks(range(len(numeric_features)))
    ax.set_xticklabels([numeric_features[i] for i in indices], rotation=90)
    ax.set_ylabel("Feature Importance")
    ax.set_title(f"{model_name.upper()} Feature Importances")
    fig.tight_layout()
    local_path = f"{model_name}_feature_importance.png"
    gcs_path = f"figures/{local_path}"
    save_and_upload_fig(fig, local_path, bucket_name, gcs_path)

def plot_loss_curve(model_stage, model_name, bucket_name):
    """GBT 或 Elastic Net 的训练 loss 曲线"""
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