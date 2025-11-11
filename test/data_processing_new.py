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


input_path = "gs://weather-2024/csv/*.csv"
output_bucket = "gs://spark-result"

spark = (
    SparkSession.builder
    .appName("WeatherProcessing")
    .config("spark.eventLog.enabled", "false")
    .getOrCreate()
)


print("Step 0: Reading files from", input_path)

df = spark.read.option("header", True).csv(input_path)
print("Step 0: Total rows read:", df.count())



# ---------
# Step 1: Delete stations with too few rows
# ---------
station_record_counts = df.groupBy("STATION").count().persist()  # Count rows per station and cache the result

# # Visualize
# counts = station_record_counts.select("count").rdd.map(lambda r: r[0]).collect()
# local_path = "/tmp/station_record_hist.png"
# local_path1 = "/tmp/station_record_hist_10000.png"

# plt.figure(figsize=(12,6))
# plt.hist(counts, bins=5000, color='skyblue', edgecolor='black')
# plt.yscale('log')
# plt.xlabel("Number of rows per station")
# plt.ylabel("Number of stations")
# plt.title("Distribution of rows per NOAA Station")
# plt.tight_layout()
# plt.savefig(local_path)
# plt.close()

# plt.figure(figsize=(12,6))
# plt.hist(counts, bins=200, range=(0,10000), color='skyblue', edgecolor='black')
# plt.xlabel("Number of rows per Station (0-10000)")
# plt.ylabel("Number of stations")
# plt.title("Distribution of rows per NOAA Station (0-10000 rows)")
# plt.tight_layout()
# plt.savefig(local_path1)
# plt.close()

thres = station_record_counts.approxQuantile("count", [0.05], 0.01)[0] 
keep_stations_df = station_record_counts.filter(F.col("count") > thres).select("STATION")
df_filtered = df.join(keep_stations_df, on="STATION", how="inner")
print("Step 1: threshold for station rows (5% quantile):", thres)

num_rows_total = station_record_counts.agg(F.sum("count")).first()[0]
num_rows_after = df_filtered.count()
num_total_csv = station_record_counts.count()
num_removed_csv = num_total_csv - keep_stations_df.count()
log_text = (
    f"Number of rows: {num_rows_total}\n"
    f"Number of rows after filtering: {num_rows_after}\n"
    f"Total number of CSV files: {num_total_csv}\n"
    f"Number of removed CSV files: {num_removed_csv}\n"
)
print(log_text)

station_record_counts.unpersist()
df.unpersist()

# # Upload plot to GCS
# from google.cloud import storage
# client = storage.Client()
# bucket = client.bucket("spark-result")
# blob = bucket.blob("station_record_hist.png")
# blob1 = bucket.blob("station_record_hist_10000.png")
# blob.upload_from_filename(local_path)
# blob1.upload_from_filename(local_path1)



# ---------
# Step 1: delete stations with too large time interval
# ---------
df_filtered = df_filtered.withColumn("DATE_TS", F.to_timestamp("DATE", "yyyy-MM-dd'T'HH:mm:ss")) 
print("Step 1: DATE_TS column added")

local_path2 = "/tmp/time_interval.png"
local_path3 = "/tmp/time_interval_180.png"

w = Window.partitionBy("STATION").orderBy(F.asc("DATE_TS"))
df_filtered = df_filtered.withColumn("dt_min", (F.unix_timestamp("DATE_TS") - F.unix_timestamp(F.lag("DATE_TS", 1).over(w)))/60) \
              .dropna(subset=["dt_min"])


min_interval, max_interval, mean_interval = df_filtered.agg(
    F.min("dt_min"), F.max("dt_min"), F.mean("dt_min")
).first()
print(f"Step 1: Before cleaning: Min: {min_interval:.2f}, Max: {max_interval:.2f}, Mean: {mean_interval:.2f}")

# Visualize
# sampled = df_filtered.select("dt_min") \
#                      .sample(0.01, seed=42) \
#                      .limit(50000) \
#                      .toPandas()

# plt.hist(sampled["dt_min"], bins=100)
# plt.savefig("/tmp/time_interval.png")
# plt.close()

# plt.hist(sampled["dt_min"], bins=100, range=(0,180))
# plt.savefig("/tmp/time_interval_180.png")
# plt.close()

thres1 = df_filtered.approxQuantile("dt_min", [0.95], 0.01)[0]
df_filtered = df_filtered.filter(F.col("dt_min") <= thres1)
after_cnt = df_filtered.count()
print("Step 1: Total stations before:", num_rows_after, "Total stations after:", after_cnt, "threshold: ", thres1)

min_i, max_i, mean_i = df_filtered.agg(F.min("dt_min"), F.max("dt_min"), F.mean("dt_min")).first()
print(f"Step 1: After cleaning - Min: {min_i:.2f}, Max: {max_i:.2f}, Mean: {mean_i:.2f}")

# # Upload plot to GCS
# from google.cloud import storage
# client = storage.Client()
# bucket = client.bucket("spark-result")
# blob = bucket.blob("time_interval.png")
# blob1 = bucket.blob("time_interval_180.png")
# blob.upload_from_filename(local_path2)
# blob1.upload_from_filename(local_path3)



# ---------
# Step 2: Select relevant columns 
# ---------
cols = ["STATION", "DATE_TS", "dt_min", "LATITUDE", "LONGITUDE", "ELEVATION", "TMP", "DEW", "WND", "VIS", "CIG", "SLP"]
df_filtered = df_filtered.select(*cols)
print("Step 2: Selected columns:", df_filtered.columns)

df_filtered = (
    df_filtered
    .withColumn("TMP", F.regexp_replace(F.split(F.col("TMP"), ",")[0], "[+]", "").cast("double") / 10)
    .withColumn("DEW", F.regexp_replace(F.split(F.col("DEW"), ",")[0], "[+]", "").cast("double") / 10)
    .withColumn("WND_Dir", F.split(F.col("WND"), ",")[0].cast("double"))
    .withColumn("WND_Speed", F.split(F.col("WND"), ",")[3].cast("double") / 10)
    .withColumn("VIS", F.split(F.col("VIS"), ",")[0].cast("double"))
    .withColumn("CIG", F.split(F.col("CIG"), ",")[0].cast("double"))
    .withColumn("SLP", F.split(F.col("SLP"), ",")[0].cast("double") / 10)
    .drop("WND")
)


# ---------
# Step 3: Delete anomaly
# ---------
abnormal_counts = df_filtered.select([
    F.sum((F.col("TMP") == 9999/10).cast("int")).alias("TMP"),
    F.sum((F.col("DEW") == 9999/10).cast("int")).alias("DEW"),
    F.sum((F.col("WND_Dir") == 999).cast("int")).alias("WND_Dir"),
    F.sum((F.col("WND_Speed") == 9999/10).cast("int")).alias("WND_Speed"),
    F.sum((F.col("VIS") == 999999).cast("int")).alias("VIS"),
    F.sum((F.col("CIG") == 99999).cast("int")).alias("CIG"),
    F.sum((F.col("SLP") == 99999/10).cast("int")).alias("SLP"),
    F.sum((F.abs(F.col("LATITUDE")) == 999.999).cast("int")).alias("LATITUDE"),
    F.sum((F.abs(F.col("LONGITUDE")) == 999.999).cast("int")).alias("LONGITUDE"),
    F.sum((F.col("ELEVATION") == 9999.9).cast("int")).alias("ELEVATION")
])
print("Step 3: Before anomaly filter, anomaly summary: ")
abnormal_counts.show(truncate=False)


# Replace sentinel values with null
df_filtered = (
    df_filtered.withColumn("TMP", F.expr("CASE WHEN TMP = 999.9 THEN NULL ELSE TMP END"))
      .withColumn("DEW", F.expr("CASE WHEN DEW = 999.9 THEN NULL ELSE DEW END"))
      .withColumn("WND_Dir", F.expr("CASE WHEN WND_Dir = 999 THEN NULL ELSE WND_Dir END"))
      .withColumn("WND_Speed", F.expr("CASE WHEN WND_Speed = 999.9 THEN NULL ELSE WND_Speed END"))
      .withColumn("VIS", F.expr("CASE WHEN VIS = 999999 THEN NULL ELSE VIS END"))
      .withColumn("CIG", F.expr("CASE WHEN CIG = 99999 THEN NULL ELSE CIG END"))
      .withColumn("SLP", F.expr("CASE WHEN SLP = 9999.9 THEN NULL ELSE SLP END"))
      .withColumn("LATITUDE", F.expr("CASE WHEN abs(LATITUDE) = 999.999 THEN NULL ELSE LATITUDE END"))
      .withColumn("LONGITUDE", F.expr("CASE WHEN abs(LONGITUDE) = 999.999 THEN NULL ELSE LONGITUDE END"))
)


# Drop rows that contain null across the weather feature set
df_filtered = df_filtered.filter(
    ~(F.col("TMP").isNull() |
      F.col("DEW").isNull() |
      F.col("WND_Dir").isNull() |
      F.col("WND_Speed").isNull() |
      F.col("VIS").isNull() |
      F.col("CIG").isNull() |
      F.col("SLP").isNull()
     )
)

# Inspect the remaining row count
print("Step 3: Remaining rows after cleaning:", df_filtered.count())
# Inspect null counts after filtering
print("Step 3: Before anomaly filter, anomaly summary: ")
abnormal_counts = df_filtered.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df_filtered.columns])
abnormal_counts.show()


# ---------
# Step 4: Insert hysteresis feature
# ---------
df_filtered = df_filtered.withColumn("TMP_lag1", F.lag("TMP", 1).over(w)) \
        .withColumn("TMP_lag2", F.lag("TMP", 2).over(w)) \
        .withColumn("DATE_TS_lag1", F.lag("DATE_TS", 1).over(w)) \
        .withColumn("DATE_TS_lag2", F.lag("DATE_TS", 2).over(w)) \
        .withColumn("speed1", (F.col("TMP") - F.col("TMP_lag1")) / F.col("dt_min")) \
        .withColumn("speed2", (F.col("TMP_lag1") - F.col("TMP_lag2")) /
                                       ((F.unix_timestamp(F.col("DATE_TS_lag1")) -
                                         F.unix_timestamp(F.col("DATE_TS_lag2"))) / 60)) \
        .withColumn("time_speed", (F.col("speed1") + F.col("speed2")) / 2) \
        .withColumn("time_a",(((F.col("TMP_lag1") - F.col("TMP_lag2")) /
              ((F.unix_timestamp(F.col("DATE_TS_lag1")) - F.unix_timestamp(F.col("DATE_TS_lag2"))) / 60)
            ) - F.col("time_speed")) / F.col("dt_min")) \
        .drop("TMP_lag2", "DATE_TS_lag1", "DATE_TS_lag2", "speed1", "speed2") \
        .dropna(subset=["TMP_lag1", "time_speed", "time_a"])

print("Step 4 finished: total rows: ", df_filtered.count())



# ---------
# Step 5: Periodic columns sin, cos
# ---------
df_filtered = df_filtered.withColumn("WND_sin", F.sin(F.radians(F.col("WND_Dir")))) \
                         .withColumn("WND_cos", F.cos(F.radians(F.col("WND_Dir")))) \
                         .drop("WND_Dir") \
                         .withColumn("year", F.year("DATE_TS")) \
                         .withColumn("month", F.month("DATE_TS")) \
                         .withColumn("day", F.dayofmonth("DATE_TS")) \
                         .withColumn("hour", F.hour("DATE_TS")) \
                         .withColumn("minute", F.minute("DATE_TS")) \
                         .withColumn("month_sin", F.sin(2 * F.pi() * F.col("month") / 12)) \
                         .withColumn("month_cos", F.cos(2 * F.pi() * F.col("month") / 12)) \
                         .withColumn("day_sin", F.sin(2 * F.pi() * F.col("day") / 31)) \
                         .withColumn("day_cos", F.cos(2 * F.pi() * F.col("day") / 31)) \
                         .withColumn("hour_sin", F.sin(2 * F.pi() * F.col("hour") / 24)) \
                         .withColumn("hour_cos", F.cos(2 * F.pi() * F.col("hour") / 24)) \
                         .withColumn("minute_sin", F.sin(2 * F.pi() * F.col("minute") / 60)) \
                         .withColumn("minute_cos", F.cos(2 * F.pi() * F.col("minute") / 60)) \
                         .drop("year", "month", "day", "hour", "minute") \
                         .withColumn("lat_sin", F.sin(F.radians(F.col("LATITUDE")))) \
                         .withColumn("lat_cos", F.cos(F.radians(F.col("LATITUDE")))) \
                         .withColumn("lon_sin", F.sin(F.radians(F.col("LONGITUDE")))) \
                         .withColumn("lon_cos", F.cos(F.radians(F.col("LONGITUDE")))) \
                         .drop("LATITUDE", "LONGITUDE") \
                         .na.drop()
print("Step 5: Periodic columns sin, cos finished")

df_filtered.write.mode("overwrite").option("header", True).parquet(os.path.join(output_bucket, "clean"))
