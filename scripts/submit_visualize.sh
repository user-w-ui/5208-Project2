set -e


CLUSTER_NAME="my-cluster"
REGION="asia-southeast1"


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

gcloud dataproc jobs submit pyspark test/visualize.py \
    --cluster=my-cluster \
    --cluster=my-cluster \
    --region=asia-southeast1 \
    --py-files=src.zip \
    -- \
    --train-path="gs://spark-result-lyx/train_withds/" \
    --test-path="gs://spark-result-lyx/test_withds/" \
    --sample-fraction=0.001 \
    --num-folds=4 \
    --bucket="spark-result"


gcloud dataproc clusters delete $CLUSTER_NAME --region=asia-southeast1 --quiet
echo "✅ Cluster deleted."
