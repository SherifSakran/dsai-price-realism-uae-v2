import os
import sys
import glob
import pandas as pd
from deployment.configs.constants import MODEL_PATH


def load_model():
    print(f"Loading artifacts from {MODEL_PATH}", file=sys.stderr)

    # Find lookup table file that starts with 'segment_lookup_table'
    lookup_pattern = os.path.join(MODEL_PATH, 'segment_lookup_table*.parquet')
    lookup_files = glob.glob(lookup_pattern)
    
    if not lookup_files:
        raise FileNotFoundError(f"No lookup table found matching pattern: {lookup_pattern}")
    
    if len(lookup_files) > 1:
        print(f"Warning: Multiple lookup tables found: {lookup_files}", file=sys.stderr)
        print(f"Using the first one: {lookup_files[0]}", file=sys.stderr)
    
    lookup_file = lookup_files[0]
    print(f"Loading lookup table: {os.path.basename(lookup_file)}", file=sys.stderr)
    lookup_table = pd.read_parquet(lookup_file)

    if 'deprecated' in lookup_table.columns:
        deprecated_count = lookup_table['deprecated'].sum()
        lookup_table = lookup_table[~lookup_table['deprecated']].copy()
        print(f"Filtered out {deprecated_count} deprecated segments (kept for reference only)", file=sys.stderr)
        print(f"Active segments: {len(lookup_table):,}", file=sys.stderr)
    else:
        print(f"Loaded lookup table with {len(lookup_table):,} segments", file=sys.stderr)

    try:
        location_tree_lookup = pd.read_parquet(os.path.join(MODEL_PATH, 'location_tree_lookup.parquet'))
        print(f"Loaded location tree lookup with {len(location_tree_lookup):,} unique locations", file=sys.stderr)
    except FileNotFoundError:
        print("Warning: location_tree_lookup.parquet not found, location rollup will be limited", file=sys.stderr)
        location_tree_lookup = pd.DataFrame()

    artifacts = {
        "lookup_table": lookup_table,
        "location_tree_lookup": location_tree_lookup
    }

    print("All artifacts loaded successfully", file=sys.stderr)

    return artifacts
