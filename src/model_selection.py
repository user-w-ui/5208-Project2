from typing import Callable, Dict, Iterable, List, Tuple

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import Evaluator
from pyspark.sql import DataFrame


def grid_search_prefix_cv(
    folds: Iterable[Tuple[DataFrame, DataFrame]],
    base_stages: List,
    estimator_builder: Callable[..., object],
    param_grid: List[Dict],
    evaluator: Evaluator,
    metric_name: str = "rmse",
) -> Tuple[Dict, List[Tuple[Dict, float]]]:
    """
    Evaluate a list of hyper-parameter dictionaries using prefix CV folds.

    Returns the best parameter set (lowest metric) and a list of all results.
    """
    results = []

    for params in param_grid:
        pipeline = Pipeline(stages=base_stages + [estimator_builder(**params)])

        fold_scores = []
        for fold_idx, (fold_train, fold_val) in enumerate(folds, 1):
            if fold_train.rdd.isEmpty() or fold_val.rdd.isEmpty():
                continue

            model = pipeline.fit(fold_train)
            preds = model.transform(fold_val)
            score = evaluator.evaluate(preds)
            fold_scores.append(score)
            print(f"Params {params} | Fold {fold_idx} {metric_name}: {score:.4f}")

        if fold_scores:
            avg_score = sum(fold_scores) / len(fold_scores)
            results.append((params, avg_score))
            print(f"Params {params} | Avg {metric_name}: {avg_score:.4f}")
        else:
            print(f"Params {params} skipped (no valid folds).")

    if not results:
        raise ValueError("No valid folds were evaluated. Check data splits.")

    results.sort(key=lambda x: x[1])  # lower is better
    best_params = results[0][0]
    print(f"Best params: {best_params} with {metric_name}: {results[0][1]:.4f}")
    return best_params, results