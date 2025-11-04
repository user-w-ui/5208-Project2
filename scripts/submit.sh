#!/bin/bash
# chmod +x submit.sh
# ./submit.sh

set -e

CLUSTER_NAME="my-cluster"
REGION="asia-southeast1"
BUCKET1="weather-2024"
BUCKET2="spark-result"
SCRIPT_PATH="gs://$BUCKET1/scripts/data_split.py"
LOG_PATH="gs://$BUCKET2/logs/data.log"

NUM_WORKERS=2
MASTER_DISK=100
WORKER_DISK=100


gcloud dataproc clusters create $CLUSTER_NAME \
  --region=$REGION \
  --num-workers=$NUM_WORKERS \
  --worker-machine-type=n2-standard-4 \
  --master-machine-type=n2-standard-4 \
  --master-boot-disk-size=$MASTER_DISK \
  --worker-boot-disk-size=$WORKER_DISK \
  --image-version="2.2-debian12" \
  --optional-components=JUPYTER \
  --enable-component-gateway \

gcloud dataproc jobs submit pyspark $SCRIPT_PATH \
  --cluster=$CLUSTER_NAME \
  --region=$REGION \
  --jars="gs://hadoop-lib/gcs/gcs-connector-latest-hadoop3.jar" \
  > >(tee /tmp/job.log) 2>&1

gsutil cp /tmp/job.log "$LOG_PATH"
echo "Logs saved to: $LOG_PATH"

# gcloud dataproc clusters delete $CLUSTER_NAME --region=$REGION --quiet
# echo "✅ Cluster deleted."
