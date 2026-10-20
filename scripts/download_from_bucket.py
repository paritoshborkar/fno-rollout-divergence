"""
Download data files from external S3 bucket. Downloaded files land
under --base-dir, defaulting to "data/raw/"

```
uv run python -m scripts.download_from_bucket --s3path <Path to data file or directory in bucket>
uv run python -m scripts.download_from_bucket --s3path training_experiment_1 \
    --base-dir outputs --bucket-name "$FNO_ROLLOUT_ARTIFACTS_BUCKET_NAME"
```
"""

import argparse
import os
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv("dev_no_docker.env")


def main(s3_path: str, base_dir: str = "data/raw", bucket_name: str | None = None):
    access_key_id = os.getenv("BUCKET_ACCESS_KEY_ID")
    secret_access_key = os.getenv("BUCKET_SECRET_ACCESS_KEY")
    endpoint_url = os.getenv("ENDPOINT_URL")

    # Falls back to the data/raw bucket's name if --bucket-name isn't given
    bucket_name = bucket_name or os.getenv("FNO_ROLLOUT_BUCKET_NAME")

    # Authenticate
    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )

    # Check if bucket exists
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except Exception as e:
        print(f"Bucket {bucket_name} not found with error: {e}")
        raise e from e

    s3_path = Path(s3_path)
    local_base_path = Path(base_dir)

    files_downloaded = 0
    for page in s3_client.get_paginator("list_objects").paginate(
        Bucket=bucket_name, Prefix=str(s3_path)
    ):
        for object in page.get("Contents", []):
            object_path = object["Key"]

            s3_client.head_object(Bucket=bucket_name, Key=object_path)

            dest_path = local_base_path / object_path
            if dest_path.exists():
                print(f"File {dest_path} exists in local, skipping download")
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                s3_client.download_file(bucket_name, str(object_path), str(dest_path))
                files_downloaded += 1
                print(f"File {dest_path} downloaded to local")

    print(f"{files_downloaded} files downloaded from bucket {bucket_name}")


if __name__ == "__main__":
    # Directory path
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3path", help="Path to data file or directory in bucket")
    parser.add_argument(
        "--base-dir",
        default="data/raw",
        help="Local directory downloaded files are placed under (default: data/raw)",
    )
    parser.add_argument(
        "--bucket-name",
        default=None,
        help="Bucket to download from (default: $FNO_ROLLOUT_BUCKET_NAME)",
    )
    cmd_args = parser.parse_args()

    main(
        s3_path=cmd_args.s3path,
        base_dir=cmd_args.base_dir,
        bucket_name=cmd_args.bucket_name,
    )
