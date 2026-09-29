"""One-off script to migrate gnn_model_test's data from a personal S3 prefix to the shared
dashboards_no_retention_rule prefix.

Usage:
    AWS_PROFILE=ds python migrate_s3_data.py            # copy only (safe, default)
    AWS_PROFILE=ds python migrate_s3_data.py --delete-source  # copy, then delete originals

Run the copy first, check the destination (e.g. via the app pointed at the new path),
then re-run with --delete-source once you're confident the copy is complete and correct.
"""

import argparse

import boto3

SOURCE_BUCKET = "mirakl-data-science-tmp2"
SOURCE_PREFIX = "nbraun/datasets/coview-mdm/"

DEST_BUCKET = "mirakl-data-science-staging"
DEST_PREFIX = "dashboards_no_retention_rule/gnn_model_test/coview-mdm/"


def list_source_keys(s3_client):
    paginator = s3_client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=SOURCE_BUCKET, Prefix=SOURCE_PREFIX):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def copy_keys(s3_client, keys):
    for i, key in enumerate(keys, start=1):
        dest_key = DEST_PREFIX + key[len(SOURCE_PREFIX) :]
        s3_client.copy_object(
            Bucket=DEST_BUCKET,
            Key=dest_key,
            CopySource={"Bucket": SOURCE_BUCKET, "Key": key},
        )
        print(
            f"[{i}/{len(keys)}] copied s3://{SOURCE_BUCKET}/{key} -> s3://{DEST_BUCKET}/{dest_key}"
        )


def delete_keys(s3_client, keys):
    for i in range(0, len(keys), 1000):
        batch = keys[i : i + 1000]
        s3_client.delete_objects(
            Bucket=SOURCE_BUCKET,
            Delete={"Objects": [{"Key": k} for k in batch]},
        )
        print(f"deleted {len(batch)} source objects (batch starting at {i})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete the original objects after copying. Off by default.",
    )
    args = parser.parse_args()

    s3_client = boto3.client("s3")

    keys = list_source_keys(s3_client)
    if not keys:
        print(f"No objects found under s3://{SOURCE_BUCKET}/{SOURCE_PREFIX}")
        return

    print(f"Found {len(keys)} objects to migrate.")
    copy_keys(s3_client, keys)
    print("Copy complete.")

    if args.delete_source:
        delete_keys(s3_client, keys)
        print("Source objects deleted.")
    else:
        print(
            "Source objects left in place. Re-run with --delete-source once verified."
        )


if __name__ == "__main__":
    main()
