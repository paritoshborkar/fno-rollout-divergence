"""
Upload NetCDF trajectory data files to external storage

```
uv run python -m scripts.upload_to_bucket --path <Path to file/dir>
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


def main(data_path: str):
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

    # Check if directory does not already exist
    path = Path(data_path)
    data_raw_path = Path("data/raw")

    # A single file uploads itself. A directory uploads every file under it.
    file_paths = (
        [path]
        if path.is_file()
        else [sub_path for sub_path in path.rglob("*") if sub_path.is_file()]
    )

    files_uploaded = 0
    for sub_path in file_paths:
        s3_path = sub_path.relative_to(data_raw_path)

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
    cmd_args = parser.parse_args()

    main(data_path=cmd_args.path)
