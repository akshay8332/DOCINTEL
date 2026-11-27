"""Copy the frozen checkpoint from S3 into the local checkpoint folder.

Used by App Runner. Local Docker Compose still mounts the folder and skips this.
Does not train or change weights.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    uri = os.environ.get("DOCINTEL_S3_CHECKPOINT_URI", "").strip()
    if not uri:
        print("DOCINTEL_S3_CHECKPOINT_URI is empty; using the local checkpoint mount.")
        return 0
    if not uri.startswith("s3://"):
        print("DOCINTEL_S3_CHECKPOINT_URI must start with s3://", file=sys.stderr)
        return 1

    checkpoint_root = Path(os.environ.get("DOCINTEL_CHECKPOINT_DIR", "/models"))
    frozen = os.environ.get("DOCINTEL_FROZEN_CHECKPOINT", "epoch-3")
    target = checkpoint_root / frozen
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "model.pt"
    if marker.is_file() and marker.stat().st_size > 0:
        print(f"Checkpoint already present at {marker}")
        return 0

    try:
        import boto3
    except ImportError:
        print("boto3 is required to download the checkpoint from S3.", file=sys.stderr)
        return 1

    # s3://bucket/prefix  -> bucket, prefix
    without = uri[len("s3://") :]
    bucket, _, prefix = without.partition("/")
    prefix = prefix.rstrip("/")
    if prefix:
        prefix = prefix + "/"

    print(f"Downloading s3://{bucket}/{prefix} -> {target}")
    client = boto3.client("s3")
    token = None
    copied = 0
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        for item in page.get("Contents") or []:
            key = item["Key"]
            if key.endswith("/"):
                continue
            relative = key[len(prefix) :] if key.startswith(prefix) else Path(key).name
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(destination))
            copied += 1
            print(f"  {key}")
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")

    if not marker.is_file():
        print(f"Download finished but {marker} is missing.", file=sys.stderr)
        return 1
    print(f"Copied {copied} objects. Ready: {marker} ({marker.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
