import os

from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from common import create_spark, load_train_test, build_feature_pipeline
from time_cv import prefix_folds
from features import infer_numeric_features, LABEL, TIMESTAMP_COL
from model_selection import grid_search_prefix_cv


def main():
    spark = create_spark("WeatherGBT")
    train_base, test = load_train_test(spark)
    numeric_features = infer_numeric_features(train_base)
    feature_pipeline, label = build_feature_pipeline(numeric_features, LABEL, normalizer="minmax")

    print("TRAIN_DATA_PATH:", os.environ.get("TRAIN_DATA_PATH"))
    print("TEST_DATA_PATH:", os.environ.get("TEST_DATA_PATH"))
    print("Training columns:", train_base.columns)
    train_base.printSchema()

    train_base = train_base.orderBy(TIMESTAMP_COL).cache()
    train_base.count()
    folds = prefix_folds(train_base, TIMESTAMP_COL, num_folds=4)

    evaluator = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="rmse")
    param_grid = [
        {"maxDepth":5, "maxIter":80, "stepSize":0.1, "maxBins":32, "subsamplingRate":1.0, "minInstancesPerNode":5},
        {"maxDepth":7, "maxIter":120, "stepSize":0.1, "maxBins":64, "subsamplingRate":0.8, "minInstancesPerNode":5},
        {"maxDepth":9, "maxIter":150, "stepSize":0.05, "maxBins":64, "subsamplingRate":0.8, "minInstancesPerNode":10},
    ]

    base_stages = list(feature_pipeline.getStages())
    estimator_builder = lambda **params: GBTRegressor(
        labelCol=label,
        featuresCol="features",
        seed=42,
        **params,
    )

    best_params, grid_results = grid_search_prefix_cv(
        folds=folds,
        base_stages=base_stages,
        estimator_builder=estimator_builder,
        param_grid=param_grid,
        evaluator=evaluator,
        metric_name="rmse",
    )

    spark.createDataFrame(
        [(str(params), score) for params, score in grid_results],
        ["params", "avg_rmse"],
    ).show(truncate=False)

    final_pipeline = Pipeline(stages=base_stages + [estimator_builder(**best_params)])
    final_model = final_pipeline.fit(train_base)
    test_rmse = evaluator.evaluate(final_model.transform(test))
    print(f"Test RMSE: {test_rmse:.4f}")
    final_model.write().overwrite().save("gs://spark-result/model_out/gbrt")


if __name__ == "__main__":
    main()
