#!/bin/bash

# Package a lookup table as model.tar.gz and upload to S3
# Usage: ./upload_model.sh <path_to_lookup_table.parquet>
# Example: ./upload_model.sh ../lookup_tables/segment_lookup_table_april_6000.parquet

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load environment variables from .env file
if [ -f "${SCRIPT_DIR}/.env" ]; then
    echo "Loading environment variables from .env file..."
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
else
    echo "Error: .env file not found in ${SCRIPT_DIR}"
    exit 1
fi

# Check for input file
LOOKUP_TABLE_PATH="$1"
if [ -z "$LOOKUP_TABLE_PATH" ]; then
    echo "Usage: ./upload_model.sh <path_to_lookup_table.parquet>"
    echo ""
    echo "Available lookup tables:"
    ls -lh "${SCRIPT_DIR}/../lookup_tables/"*.parquet 2>/dev/null || echo "  (none found in lookup_tables/)"
    exit 1
fi

if [ ! -f "$LOOKUP_TABLE_PATH" ]; then
    echo "Error: File not found: $LOOKUP_TABLE_PATH"
    exit 1
fi

# Validate MODEL_DATA_URL is set
if [ -z "$MODEL_DATA_URL" ]; then
    echo "Error: MODEL_DATA_URL not set in .env"
    exit 1
fi

REGION=${REGION:-ap-southeast-1}

echo "Lookup table: $LOOKUP_TABLE_PATH"
echo "S3 destination: $MODEL_DATA_URL"
echo ""

# Package as model.tar.gz in a temp directory
TMPDIR=$(mktemp -d)
cp "$LOOKUP_TABLE_PATH" "${TMPDIR}/segment_lookup_table.parquet"
tar -czf "${TMPDIR}/model.tar.gz" -C "${TMPDIR}" segment_lookup_table.parquet

echo "Packaged model.tar.gz ($(du -h "${TMPDIR}/model.tar.gz" | cut -f1))"

# Upload to S3
echo "Uploading to S3..."
aws s3 cp "${TMPDIR}/model.tar.gz" "$MODEL_DATA_URL" --region "$REGION"

# Cleanup
rm -rf "$TMPDIR"

echo ""
echo "Done! Model artifacts uploaded to: $MODEL_DATA_URL"
