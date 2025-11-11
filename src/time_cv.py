from pyspark.sql import functions as F
from pyspark.sql import types as T


def prefix_folds(train_df, timestamp_col, num_folds=4):
    """
    Create prefix-style folds by ordering on the timestamp and assigning a
    global index via zipWithIndex, which avoids an unpartitioned window
    operation.
    """
    sorted_df = train_df.orderBy(timestamp_col).cache()
    indexed_rdd = sorted_df.rdd.zipWithIndex().map(
        lambda x: tuple(x[0]) + (int(x[1]),)
    )
    schema = sorted_df.schema.add("_row_idx", T.LongType())
    indexed_df = sorted_df.sparkSession.createDataFrame(indexed_rdd, schema).cache()

    total = indexed_df.count()
    if total == 0:
        indexed_df.unpersist()
        sorted_df.unpersist()
        return []

    fold_size = max(total // num_folds, 1)
    folds = []
    for k in range(1, num_folds + 1):
        val_start = fold_size * (k - 1)
        val_end = fold_size * k if k < num_folds else total

        val_df = indexed_df.filter(
            (F.col("_row_idx") >= val_start) & (F.col("_row_idx") < val_end)
        ).drop("_row_idx")
        train_slice = indexed_df.filter(
            F.col("_row_idx") < val_start
        ).drop("_row_idx")

        if val_df.rdd.isEmpty():
            continue
        folds.append((train_slice, val_df))

    indexed_df.unpersist()
    sorted_df.unpersist()
    return folds
