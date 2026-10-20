"""
Upload data files to external S3 bucket. The S3 key for each
uploaded file is computed relative to --base-dir, defaulting to "data/raw".

```
uv run python -m scripts.upload_to_bucket --path <Path to file/dir>
uv run python -m scripts.upload_to_bucket --path outputs/training_experiment_1 \
    --base-dir outputs --bucket-name "$FNO_ROLLOUT_ARTIFACTS_BUCKET_NAME"
```
"""

import argparse
import os
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv("dev_no_docker.env")


def main(data_path: str, base_dir: str = "data/raw", bucket_name: str | None = None):
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

    # Check if directory does not already exist
    path = Path(data_path)
    local_base_path = Path(base_dir)

    # A single file uploads itself. A directory uploads every file under it.
    file_paths = (
        [path]
        if path.is_file()
        else [sub_path for sub_path in path.rglob("*") if sub_path.is_file()]
    )

    files_uploaded = 0
    for sub_path in file_paths:
        s3_path = sub_path.relative_to(local_base_path)

        try:
            s3_client.head_object(Bucket=bucket_name, Key=str(s3_path))
        except ClientError as e:
            # Only upload file if it does not exist in bucket
            if e.response["Error"]["Code"] == "404":
                print(f"Uploading {sub_path} to bucket {bucket_name}")
                s3_client.upload_file(str(sub_path), bucket_name, str(s3_path))
                files_uploaded += 1
            else:
                raise e from e
        else:
            print(
                f"File {sub_path} already exists in bucket {bucket_name}. Skipping upload"
            )

    print(f"{files_uploaded} files uploaded to bucket {bucket_name}")


if __name__ == "__main__":
    # Directory path
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to data file or directory")
    parser.add_argument(
        "--base-dir",
        default="data/raw",
        help="Local directory the uploaded S3 key is computed relative to "
        "(default: data/raw)",
    )
    parser.add_argument(
        "--bucket-name",
        default=None,
        help="Bucket to upload to (default: $FNO_ROLLOUT_BUCKET_NAME)",
    )
    cmd_args = parser.parse_args()

    main(
        data_path=cmd_args.path,
        base_dir=cmd_args.base_dir,
        bucket_name=cmd_args.bucket_name,
    )
