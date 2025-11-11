#!/bin/bash
# Usage:
# chmod +x submit_main.sh
# ./submit_main.sh rf   # train RandomForest
# ./submit_main.sh gbrt # train GBT

set -e

# ---------------------------
# Configuration
# ---------------------------
MODEL="$1"  # First argument: rf, gbrt, or elastic
if [[ -z "$MODEL" || ! "$MODEL" =~ ^(rf|gbrt|elastic)$ ]]; then
  echo "Usage: $0 <rf|gbrt|elastic>"
  exit 1
fi


CLUSTER_NAME="my-cluster"
REGION="asia-southeast1"
BUCKET2="spark-result-lyx"
LOG_PATH="gs://$BUCKET2/logs/main_${MODEL}.log"

TRAINSET_PATH="gs://$BUCKET2/train_withds"
TESTSET_PATH="gs://$BUCKET2/test_withds"

# NUM_WORKERS=2
# MASTER_DISK=100
# WORKER_DISK=100

# gcloud dataproc clusters create $CLUSTER_NAME \
#   --region=$REGION \
#   --num-workers=$NUM_WORKERS \
#   --worker-machine-type=n2-standard-4 \
#   --master-machine-type=n2-standard-4 \
#   --master-boot-disk-size=$MASTER_DISK \
#   --worker-boot-disk-size=$WORKER_DISK \
#   --image-version="2.2-debian12" \
#   --optional-components=JUPYTER \
#   --enable-component-gateway

zip -r src.zip ./src

gcloud dataproc jobs submit pyspark \
    test/main.py \
  --cluster=$CLUSTER_NAME \
  --region=$REGION \
  --py-files=src.zip \
  --jars="gs://hadoop-lib/gcs/gcs-connector-latest-hadoop3.jar" \
  -- \
  --model $MODEL \
  --train-path $TRAINSET_PATH \
  --test-path $TESTSET_PATH \
  --sample-fraction=0.001 \
  --num-folds 4 \
  --bucket="$BUCKET2" \
  > >(tee /tmp/job_${MODEL}.log) 2>&1

gsutil cp /tmp/job_${MODEL}.log "$LOG_PATH"
echo "Logs saved to: $LOG_PATH"

MODEL="$1"
LOCAL_PATH="models/${MODEL}_model"
BUCKET="$BUCKET2"

gsutil -m cp -r $LOCAL_PATH gs://$BUCKET/models/


gcloud dataproc clusters delete $CLUSTER_NAME --region=$REGION --quiet
echo "✅ Cluster deleted."
