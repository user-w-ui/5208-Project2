from pyspark.ml import Pipeline
from pyspark.ml.evaluation import Evaluator, RegressionEvaluator
from pyspark.sql import DataFrame
import matplotlib.pyplot as plt
from google.cloud import storage


def single_param_scan(folds, base_stages, estimator_builder, param_name, param_values, label="TMP"):
    evaluator = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="rmse")
    avg_rmses = []

    for val in param_values:
        fold_scores = []
        for fold_train, fold_val in folds:
            if fold_train.rdd.isEmpty() or fold_val.rdd.isEmpty():
                continue
            pipeline = Pipeline(stages=base_stages + [estimator_builder(**{param_name: val})])
            model = pipeline.fit(fold_train)
            preds = model.transform(fold_val)
            fold_scores.append(evaluator.evaluate(preds))
            print(f"Fold: {len(fold_scores)},{param_name}={val}, RMSE={fold_scores[-1]:.4f}")
        avg_rmse = sum(fold_scores) / len(fold_scores)
        avg_rmses.append(avg_rmse)
        print(f"{param_name}={val}, avg RMSE={avg_rmse:.4f}")

    return param_values, avg_rmses

def plot_and_upload(param_values, avg_rmses, param_name, model_name, bucket_name, local_path=None):
    plt.figure(figsize=(6,4))
    plt.plot(param_values, avg_rmses, marker='o')
    plt.xlabel(param_name)
    plt.ylabel("Average RMSE")
    plt.title(f"{model_name} - Effect of {param_name}")
    plt.grid(True)
    
    local_path = local_path or f"{model_name}_{param_name}.png"
    plt.savefig(local_path)
    plt.close()

    # upload to GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"plots/{model_name}_{param_name}.png")
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{bucket_name}/plots/")