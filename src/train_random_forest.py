from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor

from common import create_spark, load_train_test, build_feature_pipeline
from time_cv import prefix_folds
from features import infer_numeric_features, LABEL, TIMESTAMP_COL
from model_selection import grid_search_prefix_cv


def main():
    spark = create_spark("WeatherRandomForest")
    train_base, test = load_train_test(spark)
    numeric_features = infer_numeric_features(train_base)
    feature_pipeline, label = build_feature_pipeline(numeric_features, LABEL, normalizer="minmax")

    train_base = train_base.orderBy(TIMESTAMP_COL).cache()
    train_base.count()
    folds = prefix_folds(train_base, TIMESTAMP_COL, num_folds=4)

    evaluator = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="rmse")
    param_grid = [
        {"numTrees":100, "maxDepth":10, "subsamplingRate":0.8, "featureSubsetStrategy":"auto", "minInstancesPerNode":5},
        {"numTrees":200, "maxDepth":12, "subsamplingRate":0.9, "featureSubsetStrategy":"auto", "minInstancesPerNode":5},
        {"numTrees":300, "maxDepth":14, "subsamplingRate":0.7, "featureSubsetStrategy":"sqrt", "minInstancesPerNode":10},
    ]

    base_stages = list(feature_pipeline.getStages())
    estimator_builder = lambda **params: RandomForestRegressor(
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
    final_model.write().overwrite().save("models/random_forest")


if __name__ == "__main__":
    main()
