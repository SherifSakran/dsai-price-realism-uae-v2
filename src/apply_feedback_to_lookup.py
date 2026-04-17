"""
Apply CX feedback to the main lookup table (long-term fix).

This script updates the lookup table with CX-reviewed disputed listings by:
1. Fetching disputed listings from S3
2. Building feedback segment keys
3. Updating existing segments with _cx relaxation levels and feedback medians
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from io import BytesIO

sys.path.append(str(Path(__file__).parent.parent))

from configs.outlier_config import (
    CX_FEEDBACK_S3_BUCKET,
    CX_FEEDBACK_S3_KEY,
)
from configs.dataset_config import LOOKUP_TABLE_VERSION
from src.segment_outlier_detection import create_segment_key


def apply_cx_feedback_to_lookup(lookup_table: pd.DataFrame, feedback_file_path: str = None) -> pd.DataFrame:
    """
    Long-term fix: merge CX feedback into the main lookup table.
    
    For every segment_key in the feedback:
    - If the segment exists in the lookup table:
      - Update relaxation_level to '{original}_cx'
      - Update price_to_sqft_median to the feedback value
    - If the segment does not exist:
      - Skip it (do not insert new rows)
    
    Args:
        lookup_table: The combined lookup table (residential + rest)
        feedback_file_path: Optional local path to feedback parquet file (for testing).
                           If None, will fetch from S3.
    
    Returns:
        Updated lookup table with CX feedback applied
    """
    print("\n" + "="*80)
    print("Applying CX feedback (long-term fix)...")
    print("="*80)

    # Load disputed listings from local file or S3
    if feedback_file_path:
        print(f"Loading feedback from local file: {feedback_file_path}")
        try:
            disputed_df = pd.read_parquet(feedback_file_path)
            print(f"Loaded {len(disputed_df):,} disputed listings from local file")
        except Exception as e:
            print(f"Could not load feedback from local file: {e}")
            print("Skipping CX feedback adjustment")
            return lookup_table
    else:
        try:
            import boto3
            s3 = boto3.client('s3')
            print(f"Fetching s3://{CX_FEEDBACK_S3_BUCKET}/{CX_FEEDBACK_S3_KEY}")
            resp = s3.get_object(Bucket=CX_FEEDBACK_S3_BUCKET, Key=CX_FEEDBACK_S3_KEY)
            body = resp['Body'].read()
            disputed_df = pd.read_parquet(BytesIO(body))
            print(f"Loaded {len(disputed_df):,} disputed listings from S3")
        except Exception as e:
            print(f"Could not load CX feedback from S3: {e}")
            print("Skipping CX feedback adjustment")
            return lookup_table

    if disputed_df.empty:
        print("No disputed listings found, skipping CX feedback adjustment")
        return lookup_table

    lookup_table = lookup_table.copy()

    # Build feedback segment keys from disputed listings
    feedback_segments = {}  # {segment_key: [price_to_sqft values]}

    for _, listing in disputed_df.iterrows():
        price = pd.to_numeric(listing.get('price', None), errors='coerce')
        sqft = pd.to_numeric(listing.get('property_sqft', None), errors='coerce')

        if pd.isna(price) or pd.isna(sqft) or sqft == 0:
            continue

        price_to_sqft = price / sqft

        row = pd.Series({
            'location_id': str(listing.get('location_id', '')),
            'housing_type_name': str(listing.get('housing_type_name', listing.get('property_type', ''))),
            'offering_type_name': str(listing.get('offering_type_name', '')),
            'property_price_type_name': str(listing.get('property_price_type_name', '')),
            'bedrooms': str(listing.get('bedrooms', '')),
        })

        # Generate both with- and without-bedrooms keys
        for include_bed in [True, False]:
            sk = create_segment_key(row, include_bedrooms=include_bed, location_level='location_id')
            if sk not in feedback_segments:
                feedback_segments[sk] = []
            feedback_segments[sk].append(price_to_sqft)

    if not feedback_segments:
        print("No valid feedback segments built, skipping")
        return lookup_table

    # Collapse to median per segment
    feedback_medians = {sk: float(np.median(vals)) for sk, vals in feedback_segments.items()}

    # Add 'deprecated' column if it doesn't exist
    if 'deprecated' not in lookup_table.columns:
        lookup_table['deprecated'] = False

    new_segments = []
    updated_count = 0
    skipped_count = 0

    for segment_key, feedback_pts_median in feedback_medians.items():
        mask = lookup_table['segment_key'] == segment_key
        if mask.any():
            # Flag the original segment as deprecated (for reference only)
            idx = lookup_table[mask].index[0]
            original_level = lookup_table.loc[idx, 'relaxation_level']
            lookup_table.loc[idx, 'deprecated'] = True
            
            # Create a new _cx segment row with updated price_to_sqft_median and nulls for other values
            new_segment = {
                'segment_key': segment_key,
                'segment_count': None,
                'relaxation_level': f"{original_level}_cx",
                'price_to_sqft_median': feedback_pts_median,
                'deprecated': False,
                # All other statistical columns are set to None/NaN
                'price_to_sqft_q1': None,
                'price_to_sqft_q3': None,
                'price_to_sqft_mean': None,
                'price_to_sqft_std': None,
                'price_q1': None,
                'price_q3': None,
                'price_median': None,
                'price_mean': None,
                'price_std': None,
                'property_sqft_q1': None,
                'property_sqft_q3': None,
                'property_sqft_median': None,
                'property_sqft_mean': None,
                'property_sqft_std': None,
            }
            new_segments.append(new_segment)
            updated_count += 1
            print(f"  Created _cx segment for {segment_key}: pts_median={feedback_pts_median:.2f}")
        else:
            # Segment not in lookup table — skip. A disputed listing must have
            # been flagged (unseen=False) originally, so its segment should exist.
            # If the table was rebuilt and the segment no longer exists, we do not
            # insert it — an unknown segment defaults to unseen=True and the
            # listing would pass through without bounds, which is the safer default.
            skipped_count += 1

    # Append new _cx segments to the lookup table
    if new_segments:
        new_segments_df = pd.DataFrame(new_segments)
        lookup_table = pd.concat([lookup_table, new_segments_df], ignore_index=True)
    
    print(f"CX feedback applied: {updated_count} new _cx segments created, {skipped_count} segments skipped (not in lookup)")
    print(f"Total segments after feedback: {len(lookup_table):,} ({updated_count} deprecated, {updated_count} new _cx)")

    return lookup_table


if __name__ == "__main__":
    """
    Standalone script to apply CX feedback to an existing lookup table.
    
    Reads configuration from dataset_config.py and uses local disputed listings file.
    """
    # Define paths using config values
    input_table_path = f'../lookup_tables/segment_lookup_table_{LOOKUP_TABLE_VERSION}.parquet'
    output_table_path = f'../lookup_tables/segment_lookup_table_{LOOKUP_TABLE_VERSION}_with_feedback.parquet'
    feedback_file_path = '../disputed_listings/disputed_listings_details.parquet'
    
    print("=" * 80)
    print("APPLYING CX FEEDBACK TO LOOKUP TABLE")
    print("=" * 80)
    print(f"Lookup table version: {LOOKUP_TABLE_VERSION}")
    print(f"Input table: {input_table_path}")
    print(f"Output table: {output_table_path}")
    print(f"Feedback file: {feedback_file_path}")
    print()
    
    print(f"Loading lookup table from: {input_table_path}")
    lookup_table = pd.read_parquet(input_table_path)
    print(f"Loaded {len(lookup_table):,} segments")
    
    updated_table = apply_cx_feedback_to_lookup(lookup_table, feedback_file_path=feedback_file_path)
    
    print(f"\nSaving updated lookup table to: {output_table_path}")
    updated_table.to_parquet(output_table_path, index=False)
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
