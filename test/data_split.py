from pyspark.sql import SparkSession, functions as F, Window
from pyspark.sql.types import (
    StructType, StructField, StringType, TimestampType, DoubleType,
)
from pyspark.storagelevel import StorageLevel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import reduce
import os


input_path = "gs://spark-result/clean"
output_bucket = "gs://spark-result"

spark = SparkSession.builder.appName("WeatherProcessing").getOrCreate()

print("Step: Reading files from", input_path)
df_filtered = spark.read.parquet(input_path)


# ---------
# Step 6: Filter TMP
# ---------
lower, upper = df_filtered.approxQuantile("TMP", [0.05, 0.95], 0.01)
print(f"Step 6: Lower threshold: {lower}, Upper threshold: {upper}")

df_filtered = df_filtered.filter((df_filtered["TMP"] >= lower) & (df_filtered["TMP"] <= upper))


# ---------
# Step 7: Train/test split
# ---------
min_date, max_date = df_filtered.agg(
    F.min("DATE_TS").alias("min"),
    F.max("DATE_TS").alias("max")
).first()

cutoff = min_date + (max_date - min_date) * 0.7

train_df = df_filtered.filter(F.col("DATE_TS") <= cutoff)
test_df  = df_filtered.filter(F.col("DATE_TS") > cutoff)
print("Step 7: Train count:", train_df.count(), "Test count:", test_df.count())

train_df.write.mode("overwrite").parquet(os.path.join(output_bucket, "train_withds1"))
test_df.write.mode("overwrite").parquet(os.path.join(output_bucket, "test_withds1"))
print("All steps completed. CSVs saved to", output_bucket)

