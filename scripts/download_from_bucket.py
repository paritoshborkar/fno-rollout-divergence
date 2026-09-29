"""
Download data files from external storage
```

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


def main(s3_path: str):
    access_key_id = os.getenv("BUCKET_ACCESS_KEY_ID")
    secret_access_key = os.getenv("BUCKET_SECRET_ACCESS_KEY")
    endpoint_url = os.getenv("ENDPOINT_URL")

    # Read bucket name and endpoint from env file
    bucket_name = os.getenv("FNO_ROLLOUT_BUCKET_NAME")

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
    data_raw_path = Path("data/raw")

    files_downloaded = 0
    for page in s3_client.get_paginator("list_objects").paginate(
        Bucket=bucket_name, Prefix=str(s3_path)
    ):
        for object in page.get("Contents", []):
            object_path = object["Key"]

            s3_client.head_object(Bucket=bucket_name, Key=object_path)

            dest_path = data_raw_path / object_path
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
    cmd_args = parser.parse_args()

    main(s3_path=cmd_args.s3path)
