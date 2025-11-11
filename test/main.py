import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

import sys, os, time
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.time_cv import prefix_folds
from src.model_selection import grid_search_prefix_cv
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
    parser.add_argument("--sample-fraction", type=float, default=0.01)
    parser.add_argument("--num-folds", type=int, default=4)
    parser.add_argument("--bucket", default="spark-result")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("WeatherForecast").getOrCreate()
    train_df = spark.read.parquet(args.train_path).drop("STATION") 
    train_df = train_df.withColumn("ELEVATION", train_df["ELEVATION"].cast(DoubleType()))
    test_df = spark.read.parquet(args.test_path).drop("STATION") 
    test_df = test_df.withColumn("ELEVATION", test_df["ELEVATION"].cast(DoubleType()))

    train_sorted = train_df.orderBy(TIMESTAMP_COL)
    total_rows = train_sorted.count()
    # Sample evenly spaced rows based on the requested fraction
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
        {"numTrees": 80,  "maxDepth": 8,  "subsamplingRate": 0.8},
        # {"numTrees":160,  "maxDepth": 8,  "subsamplingRate": 0.8},
        # {"numTrees":120,  "maxDepth":10, "subsamplingRate": 0.9},
        # {"numTrees":240,  "maxDepth":10, "subsamplingRate": 0.9},
        # {"numTrees":200,  "maxDepth":14, "subsamplingRate": 1.0},
        # {"numTrees":320,  "maxDepth":14, "subsamplingRate": 1.0},
        ]
        estimator_builder = lambda **p: RandomForestRegressor(labelCol=LABEL, featuresCol="features", seed=42, **p)
    elif args.model == "gbrt":
        param_grid = [
            {"maxDepth":5, "maxIter":80, "stepSize":0.1, "maxBins":32, "subsamplingRate":1.0, "minInstancesPerNode":5},
            {"maxDepth":7, "maxIter":120, "stepSize":0.1, "maxBins":64, "subsamplingRate":0.8, "minInstancesPerNode":5},
        ]
        estimator_builder = lambda **p: GBTRegressor(labelCol=LABEL, featuresCol="features", seed=42, **p)
    elif args.model == "elastic":
        param_grid = [
            # First Try
            # {"regParam": 0.00025, "elasticNetParam": 0.1, "maxIter": 300},
            # {"regParam": 0.01,  "elasticNetParam": 0.5, "maxIter": 400},
            # {"regParam": 0.03,  "elasticNetParam": 0.8, "maxIter": 400},
            # {"regParam": 0.08, "elasticNetParam": 1.0, "maxIter": 500},
            # Second Try
            {"regParam": 0.00015, "elasticNetParam": 0.08, "maxIter": 80,  "tol": 1e-6},
            # {"regParam": 0.00025, "elasticNetParam": 0.10, "maxIter": 80,  "tol": 1e-6},
            # {"regParam": 0.00040, "elasticNetParam": 0.15, "maxIter": 100, "tol": 1e-6},
            # Third Try
            {"regParam": 0.00018, "elasticNetParam": 0.09, "maxIter": 160,  "tol": 5e-7},
            {"regParam": 0.00012, "elasticNetParam": 0.07, "maxIter": 120,  "tol": 1e-6},
            {"regParam": 0.0003,"elasticNetParam": 0.12, "maxIter": 200, "tol": 1e-6},
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
    print("best params: ", best_params)
    spark.createDataFrame(
        [(str(params), score) for params, score in grid_results],
        ["params", "avg_rmse"],
        ).show(truncate=False)

    final_pipeline = Pipeline(stages=base_stages + [estimator_builder(**best_params)])
    # Record training time
    start_time = time.time()
    final_model = final_pipeline.fit(train_df)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    if args.model == "elastic":
        lin_stage = final_model.stages[-1]
        if hasattr(lin_stage, "coefficients"):
            print("Elastic Net coefficients (scaled feature space):")
            for feature_name, coef in zip(numeric_features, lin_stage.coefficients.toArray()):
                print(f"  {feature_name}: {coef}")
            print(f"Intercept: {lin_stage.intercept}")

    start_time = time.time()
    preds = final_model.transform(test_df)
    end_time = time.time()
    print(f"Prediction time: {end_time - start_time:.2f} seconds")


    test_rmse = evaluator.evaluate(preds)
    test_mae  = evaluator_mae.evaluate(preds)
    test_r2   = evaluator_r2.evaluate(preds)
    print(f"Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}")


    # Residual plot
    plot.plot_residuals(preds, LABEL, args.model, args.bucket)

    # Scatter plot
    plot.plot_pred_vs_actual(preds, LABEL, args.model, args.bucket)

    # Time-series plot
    plot.plot_time_series(preds, test_df, LABEL, TIMESTAMP_COL, args.model, args.bucket)

    # Feature importance plot
    if args.model in ["rf", "gbrt"]:
        plot.plot_feature_importances(final_model.stages[-1], numeric_features, args.model, args.bucket)

    # Loss curve
    if args.model in ["gbrt", "elastic"]:
        plot.plot_loss_curve(final_model.stages[-1], args.model, args.bucket)


    from google.cloud import storage

    local_model_path = Path("models") / f"{args.model}_model"
    local_model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.write().overwrite().save(f"file://{local_model_path.resolve()}")
    print(f"Model saved locally at {local_model_path.resolve()}")

    gcs_model_path = f"models/{args.model}_model"

    client = storage.Client()
    bucket = client.bucket(args.bucket)
    for root, dirs, files in os.walk(local_model_path):
        for file in files:
            local_file = Path(root) / file
            relative_path = os.path.relpath(local_file, local_model_path)
            blob = bucket.blob(f"{gcs_model_path}/{relative_path}")
            blob.upload_from_filename(str(local_file))

    print(f"Model uploaded to gs://{args.bucket}/{gcs_model_path}")
    print(f"Model !")



if __name__ == "__main__":
    main()
