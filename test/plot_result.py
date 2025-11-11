import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import plot

numeric_features = [
        "dt_min", "ELEVATION", "DEW", "VIS", "CIG", "SLP", "WND_Speed",
        "TMP_lag1", "time_speed", "time_a", "WND_sin", "WND_cos",
        "month_sin", "month_cos", "day_sin", "day_cos",
        "hour_sin", "hour_cos", "minute_sin", "minute_cos",
        "lat_sin", "lat_cos", "lon_sin", "lon_cos"
    ]
LABEL = "TMP"
TIMESTAMP_COL = "DATE_TS"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "gbrt", "elastic"], required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--bucket", default="spark-result")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("WeatherForecast").getOrCreate()
    train_df = spark.read.parquet(args.train_path).drop("STATION") 
    train_df = train_df.withColumn("ELEVATION", train_df["ELEVATION"].cast(DoubleType()))
    test_df = spark.read.parquet(args.test_path).drop("STATION") 
    test_df = test_df.withColumn("ELEVATION", test_df["ELEVATION"].cast(DoubleType()))

    # ✅ Fill in the model best params
    if args.model == "rf":
        best_params = {
            "numTrees": 100,
            "maxDepth": 8,
            "subsamplingRate": 0.9,
            "minInstancesPerNode": 5,
        }
        estimator_builder = RandomForestRegressor(labelCol=LABEL, featuresCol="features", seed=42, **best_params)

    elif args.model == "gbrt":
        best_params = { 
            "maxDepth": 4,
            "maxIter": 70,
            "stepSize": 0.2,
            "subsamplingRate": 1.0,
            "minInstancesPerNode": 5
        }
        estimator_builder = GBTRegressor(labelCol=LABEL, featuresCol="features", seed=42, **best_params)

    elif args.model == "elastic":
        best_params = {
            "regParam": 0.00012,
            "elasticNetParam": 0.07,
            "maxIter": 120,
            "tol": 1e-6
        }
        estimator_builder = LinearRegression(
            labelCol=LABEL,
            featuresCol="features",
            solver="auto",
            standardization=False,
            fitIntercept=True,
            **best_params,
        )


    assembler = VectorAssembler(inputCols=numeric_features, outputCol="features_raw")
    scaler = MinMaxScaler(inputCol="features_raw", outputCol="features")
    pipeline = Pipeline(stages=[assembler, scaler, estimator_builder])

    print("Start Training...")
    start_time = time.time()
    final_model = pipeline.fit(train_df)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    ### ===== metrics =====
    evaluator = RegressionEvaluator(labelCol=LABEL, predictionCol="prediction", metricName="rmse")
    evaluator_mae  = RegressionEvaluator(labelCol=LABEL, predictionCol="prediction", metricName="mae")
    evaluator_r2   = RegressionEvaluator(labelCol=LABEL, predictionCol="prediction", metricName="r2")

    def compute_mape(df):
        safe_div = F.when(F.abs(F.col(LABEL)) > 1e-6,
                          F.abs((F.col(LABEL) - F.col("prediction")) / F.col(LABEL)))
        return df.select(F.mean(safe_div).alias("mape")).collect()[0]["mape"]

    ### ===== Train metrics =====
    start_time = time.time()
    train_preds = final_model.transform(train_df)
    end_time = time.time()
    print(f"Prediction time (train set): {end_time - start_time:.2f} seconds")
    train_rmse = evaluator.evaluate(train_preds)
    train_mae  = evaluator_mae.evaluate(train_preds)
    train_r2   = evaluator_r2.evaluate(train_preds)
    train_mape = compute_mape(train_preds)
    print(f"[Train] RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, MAPE: {train_mape:.4f}, R2: {train_r2:.4f}")

    ### ===== Test metrics =====
    start_time = time.time()
    preds = final_model.transform(test_df)
    end_time = time.time()
    print(f"Prediction time (test set): {end_time - start_time:.2f} seconds")
    test_rmse = evaluator.evaluate(preds)
    test_mae  = evaluator_mae.evaluate(preds)
    test_r2   = evaluator_r2.evaluate(preds)
    test_mape = compute_mape(preds)
    print(f"[Test] RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.4f}, R2: {test_r2:.4f}")

    ### ===== Sample and Plot =====
    def downsample(df, limit=10000):
        cnt = df.count()
        if cnt > limit:
            return df.sample(False, limit / cnt, seed=42)
        return df

    sampled_preds = downsample(preds, 10000)

    plot.plot_residuals(sampled_preds, LABEL, args.model, args.bucket)
    plot.plot_pred_vs_actual(sampled_preds, LABEL, args.model, args.bucket)
    plot.plot_time_series(sampled_preds, LABEL, TIMESTAMP_COL, args.model, args.bucket)

    print("[Feature Importance] model_stage:", final_model.stages[-1])
    print("[Feature Importance] numeric_features:", numeric_features)
    if args.model in ["rf", "gbrt"]:
        plot.plot_feature_importances(final_model.stages[-1], numeric_features, args.model, args.bucket)

    # Loss curve
    if args.model in ["elastic"]:
        plot.plot_loss_curve(final_model.stages[-1], args.model, args.bucket)
    
    print("Finished!")

    # from google.cloud import storage
    # import os

    # local_model_path = f"models/{args.model}_model"
    # final_model.write().overwrite().save(local_model_path)
    # gcs_model_path = f"models/{args.model}_model"

    # client = storage.Client()
    # bucket = client.bucket(args.bucket)
    # for root, dirs, files in os.walk(local_model_path):
    #     for file in files:
    #         local_file = os.path.join(root, file)
    #         relative_path = os.path.relpath(local_file, local_model_path)
    #         blob = bucket.blob(f"{gcs_model_path}/{relative_path}")
    #         blob.upload_from_filename(local_file)

    # print(f"Model uploaded to gs://{args.bucket}/{gcs_model_path}")
    # print(f"Model !")



if __name__ == "__main__":
    main()
