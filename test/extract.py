import tarfile
from google.cloud import storage
import tempfile, os

client = storage.Client()
bucket = client.bucket("weather-2024")

tmp_dir = tempfile.mkdtemp()
tar_path = os.path.join(tmp_dir, "2024.tar.gz")

# 1. Download the archive
os.system(f"gsutil cp gs://weather-2024/2024.tar.gz {tar_path}")

# 2. Stream the archive and upload files without persisting to disk
with tarfile.open(tar_path, "r:gz") as tar:
    for member in tar:
        if member.isfile() and member.name.endswith(".csv"):
            f = tar.extractfile(member)
            blob = bucket.blob(f"csv/{os.path.basename(member.name)}")
            blob.upload_from_file(f, rewind=True)
            f.close()

# 3. Remove the temporary archive
os.remove(tar_path)
