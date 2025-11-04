import argparse
from common import create_spark

parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True)
args, _ = parser.parse_known_args()

spark = create_spark("SchemaInspect")
df = spark.read.parquet(args.path)
df.printSchema()
df.show(5)
