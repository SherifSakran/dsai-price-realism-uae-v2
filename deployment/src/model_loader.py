import sys
import threading
from datetime import datetime
from io import BytesIO
from typing import Optional
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from deployment.configs.constants import (
    LOOKUP_TABLE_S3_BUCKET,
    LOOKUP_TABLE_S3_KEY,
    LOCATION_TREE_S3_KEY,
    LOOKUP_REFRESH_SECONDS,
)


artifacts = {
    "lookup_table": pd.DataFrame(),
    "location_tree_lookup": pd.DataFrame(),
}
lookup_last_update_ts: Optional[datetime] = None
_lookup_lock = threading.Lock()
_lookup_updating = False


def _load_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
    s3 = boto3.client('s3')
    resp = s3.get_object(Bucket=bucket, Key=key)
    body = resp['Body'].read()
    return pd.read_parquet(BytesIO(body))


def load_model():
    global artifacts, lookup_last_update_ts

    print(f"[LOOKUP] Fetching s3://{LOOKUP_TABLE_S3_BUCKET}/{LOOKUP_TABLE_S3_KEY}", file=sys.stderr)
    lookup_table = _load_parquet_from_s3(LOOKUP_TABLE_S3_BUCKET, LOOKUP_TABLE_S3_KEY)

    if 'deprecated' in lookup_table.columns:
        deprecated_count = lookup_table['deprecated'].sum()
        lookup_table = lookup_table[~lookup_table['deprecated']].copy()
        print(f"[LOOKUP] Filtered out {deprecated_count} deprecated segments (kept for reference only)", file=sys.stderr)
        print(f"[LOOKUP] Active segments: {len(lookup_table):,}", file=sys.stderr)
    else:
        print(f"[LOOKUP] Loaded lookup table with {len(lookup_table):,} segments", file=sys.stderr)

    try:
        print(f"[LOOKUP] Fetching s3://{LOOKUP_TABLE_S3_BUCKET}/{LOCATION_TREE_S3_KEY}", file=sys.stderr)
        location_tree_lookup = _load_parquet_from_s3(LOOKUP_TABLE_S3_BUCKET, LOCATION_TREE_S3_KEY)
        print(f"[LOOKUP] Loaded location tree lookup with {len(location_tree_lookup):,} unique locations", file=sys.stderr)
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code in ('NoSuchKey', '404'):
            print("[LOOKUP] Warning: location_tree_lookup.parquet not found in S3, location rollup will be limited", file=sys.stderr)
            location_tree_lookup = pd.DataFrame()
        else:
            raise

    new_artifacts = {
        "lookup_table": lookup_table,
        "location_tree_lookup": location_tree_lookup,
    }

    with _lookup_lock:
        artifacts = new_artifacts
        lookup_last_update_ts = datetime.now()

    print("[LOOKUP] All artifacts loaded successfully from S3", file=sys.stderr)
    return artifacts


def _background_refresh_lookup():
    global _lookup_updating
    try:
        load_model()
    except Exception as e:
        print(f"[LOOKUP] Background refresh failed: {e}, keeping current artifacts", file=sys.stderr)
    finally:
        _lookup_updating = False


def maybe_refresh_lookup():
    global _lookup_updating
    if lookup_last_update_ts is None:
        return
    elapsed = (datetime.now() - lookup_last_update_ts).total_seconds()
    if elapsed >= LOOKUP_REFRESH_SECONDS and not _lookup_updating:
        _lookup_updating = True
        print("[LOOKUP] Lookup table stale, triggering background refresh", file=sys.stderr)
        thread = threading.Thread(target=_background_refresh_lookup, daemon=True)
        thread.start()
