import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.time_cv import prefix_folds
from src.model_selection import grid_search_prefix_cv
from src import plot

# numeric_features = [
#         "dt_min", "ELEVATION", "DEW", "VIS", "CIG", "SLP", "WND_Speed",
#         "TMP_lag1", "time_speed", "time_a", "WND_sin", "WND_cos",
#         "month_sin", "month_cos", "day_sin", "day_cos",
#         "hour_sin", "hour_cos", "minute_sin", "minute_cos",
#         "lat_sin", "lat_cos", "lon_sin", "lon_cos"
#     ]
numeric_features = ["dt_min"]
LABEL = "TMP"
TIMESTAMP_COL = "DATE_TS"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "gbrt"], required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--sample-fraction", type=float, default=0.01)
    parser.add_argument("--num-folds", type=int, default=4)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("WeatherForecast").getOrCreate()
    train_df = spark.read.parquet(args.train_path).drop("STATION") 
    train_df = train_df.withColumn("ELEVATION", train_df["ELEVATION"].cast(DoubleType())).limit(1000)
    test_df = spark.read.parquet(args.test_path).drop("STATION") 
    test_df = test_df.withColumn("ELEVATION", test_df["ELEVATION"].cast(DoubleType())).limit(50)

    train_sorted = train_df.orderBy(TIMESTAMP_COL)
    total_rows = train_sorted.count()
    # 按 fraction 等距抽
    step = int(1 / args.sample_fraction)

    train_sample = train_sorted.rdd.zipWithIndex() \
        .filter(lambda x: x[1] % step == 0) \
        .map(lambda x: x[0]) \
        .toDF(train_df.schema)
    print(f"Total rows: {total_rows}")
    print(f"Sampled rows (stride): {train_sample.count()}")
    train_sample = train_sample.cache()
    folds = prefix_folds(train_sample, TIMESTAMP_COL, num_folds=args.num_folds)

    evaluator = RegressionEvaluator(labelCol=LABEL, predictionCol="prediction", metricName="rmse")
    evaluator_mae  = RegressionEvaluator(labelCol=LABEL, predictionCol="prediction", metricName="mae")
    evaluator_r2   = RegressionEvaluator(labelCol=LABEL, predictionCol="prediction", metricName="r2")


    assembler = VectorAssembler(inputCols=numeric_features, outputCol="features_raw")
    scaler = MinMaxScaler(inputCol="features_raw", outputCol="features")
    # scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    base_stages = [assembler, scaler]

    if args.model == "rf":
        param_grid = [
            {"numTrees":100, "maxDepth":10, "subsamplingRate":0.8, "featureSubsetStrategy":"auto", "minInstancesPerNode":5},
            {"numTrees":200, "maxDepth":12, "subsamplingRate":0.9, "featureSubsetStrategy":"auto", "minInstancesPerNode":5},
        ]
        estimator_builder = lambda **p: RandomForestRegressor(labelCol=LABEL, featuresCol="features", seed=42, **p)
    elif args.model == "gbtr":
        param_grid = [
            {"maxDepth":5, "maxIter":80, "stepSize":0.1, "maxBins":32, "subsamplingRate":1.0, "minInstancesPerNode":5},
            {"maxDepth":7, "maxIter":120, "stepSize":0.1, "maxBins":64, "subsamplingRate":0.8, "minInstancesPerNode":5},
        ]
        estimator_builder = lambda **p: GBTRegressor(labelCol=LABEL, featuresCol="features", seed=42, **p)
    elif args.model == "lr":
        param_grid = [
            {"regParam": 0.01, "elasticNetParam": 0.0, "maxIter": 200},
            {"regParam": 0.1,  "elasticNetParam": 0.5, "maxIter": 200},
            {"regParam": 0.2,  "elasticNetParam": 0.9, "maxIter": 300},
        ]
        estimator_builder = lambda **p: LinearRegression(
            labelCol=LABEL,
            featuresCol="features",
            solver="auto",
            standardization=False,
            fitIntercept=True,
            **p,
        )


    best_params, grid_results = grid_search_prefix_cv(folds, base_stages, estimator_builder, param_grid, evaluator)

    final_pipeline = Pipeline(stages=base_stages + [estimator_builder(**best_params)])
    # 训练前记录时间
    start_time = time.time()
    final_model = final_pipeline.fit(train_df)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    start_time = time.time()
    preds = final_model.transform(test_df)
    end_time = time.time()
    print(f"Prediction time: {end_time - start_time:.2f} seconds")


    test_rmse = evaluator.evaluate(preds)
    test_mae  = evaluator_mae.evaluate(preds)
    test_r2   = evaluator_r2.evaluate(preds)
    print(f"Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}")


    # 残差图
    plot.plot_residuals(preds, LABEL, args.model, bucket_name)

    # 散点图
    plot.plot_pred_vs_actual(preds, LABEL, args.model, bucket_name)

    # 时间序列图
    plot.plot_time_series(preds, test_df, LABEL, TIMESTAMP_COL, args.model, bucket_name)

    # 特征重要性
    if args.model in ["rf", "gbtr"]:
        plot.plot_feature_importances(final_model.stages[-1], numeric_features, args.model, bucket_name)

    # Loss curve
    if args.model in ["gbtr", "lr"]:
        plot.plot_loss_curve(final_model.stages[-1], args.model, bucket_name)

    from google.cloud import storage
    import os

    local_model_path = f"models/{args.model}_model"
    final_model.write().overwrite().save(local_model_path)
    bucket_name = "spark-result"  
    gcs_model_path = f"models/{args.model}_model"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for root, dirs, files in os.walk(local_model_path):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, local_model_path)
            blob = bucket.blob(f"{gcs_model_path}/{relative_path}")
            blob.upload_from_filename(local_file)

    print(f"Model uploaded to gs://{bucket_name}/{gcs_model_path}")




if __name__ == "__main__":
    main()
