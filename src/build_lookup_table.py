"""
Build lookup table for segment-based outlier detection.

This script creates a lookup table containing segment definitions and statistics
for price, property_sqft, and price_to_sqft for top residential housing types.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from configs.outlier_config import (
    TOP_RESIDENTIAL_TYPES,
    REST_OF_HOUSING_TYPES,
    MIN_SEGMENT_SIZE,
    MIN_SEGMENT_SIZE_REST,
    OFFERING_TYPE_ID_TO_NAME,
    HOUSING_TYPE_ID_MAP,
    PRICE_TYPE_MAP,
)
from configs.dataset_config import (
    DATA_DATE_SUFFIX
)
from src.segment_outlier_detection import (
    assign_segments_with_relaxation,
    create_segment_key
)

version = DATA_DATE_SUFFIX

def build_lookup_table(df: pd.DataFrame, 
                       min_segment_sizes: dict = None,
                       housing_types_filter: list = None) -> pd.DataFrame:
    """
    Build lookup table with segment statistics.
    
    Args:
        df: DataFrame with listings (must have required columns)
        min_segment_sizes: Dictionary mapping housing types to minimum segment sizes
        housing_types_filter: List of housing types to process (None = all)
    
    Returns:
        DataFrame with segment-level statistics (lookup table)
    """
    if min_segment_sizes is None:
        min_segment_sizes = {**MIN_SEGMENT_SIZE, **MIN_SEGMENT_SIZE_REST}
    
    # Filter by housing types if specified
    if housing_types_filter is not None:
        df = df[df['housing_type_name'].isin(housing_types_filter)].copy()
    
    # Convert to float to handle Decimal types
    df = df.copy()
    for attr in ['price', 'property_sqft', 'price_to_sqft']:
        df[attr] = pd.to_numeric(df[attr], errors='coerce')
    
    # Assign segments with relaxation
    print("Assigning segments with relaxation...")
    start_time = time.time()
    df = assign_segments_with_relaxation(df, min_segment_sizes)
    assign_time = time.time() - start_time
    print(f"Segment assignment completed in {assign_time:.2f} seconds ({assign_time/60:.2f} minutes)")
    
    # Filter to only listings with valid segments
    df_with_segments = df[df['segment_key'].notna()].copy()
    
    print(f"Listings with segments: {len(df_with_segments):,}")
    print(f"Unique segments: {df_with_segments['segment_key'].nunique():,}")
    
    # Calculate statistics per segment
    print("\nCalculating segment statistics...")
    stats_start_time = time.time()
    
    # For without_bedrooms segments, we need to compute stats using ALL listings
    # that match the location/housing/offering/price, not just those assigned to it
    
    # First, identify all unique segment keys
    all_segments = df_with_segments['segment_key'].unique()
    
    lookup_rows = []
    
    for segment_key in all_segments:
        # Get listings assigned to this segment
        segment_df = df_with_segments[df_with_segments['segment_key'] == segment_key]
        relaxation_level = segment_df['relaxation_level'].iloc[0]
        
        # If this is a without_bedrooms segment, compute stats from ALL listings
        # that match the base criteria (location/housing/offering/price)
        if relaxation_level == 'without_bedrooms':
            # Parse segment key to get base criteria
            parts = segment_key.split('|')
            if len(parts) >= 4:
                location_id = parts[0]
                housing_type = parts[1]
                offering_type = parts[2]
                price_type = parts[3]
                
                # Find ALL listings matching these criteria (including with_bedrooms segments)
                base_mask = (
                    (df_with_segments['location_id'].astype(str) == location_id) &
                    (df_with_segments['housing_type_name'] == housing_type) &
                    (df_with_segments['offering_type_name'] == offering_type) &
                    (df_with_segments['property_price_type_name'] == price_type)
                )
                stats_df = df_with_segments[base_mask]
            else:
                stats_df = segment_df
        else:
            # For with_bedrooms segments, use only assigned listings
            stats_df = segment_df
        
        # Compute statistics
        row = {
            'segment_key': segment_key,
            'segment_count': len(stats_df),
            'relaxation_level': relaxation_level,
            
            # Price to sqft stats
            'price_to_sqft_q1': stats_df['price_to_sqft'].quantile(0.25),
            'price_to_sqft_q3': stats_df['price_to_sqft'].quantile(0.75),
            'price_to_sqft_median': stats_df['price_to_sqft'].median(),
            'price_to_sqft_mean': stats_df['price_to_sqft'].mean(),
            'price_to_sqft_std': stats_df['price_to_sqft'].std(),
            
            # Price stats
            'price_q1': stats_df['price'].quantile(0.25),
            'price_q3': stats_df['price'].quantile(0.75),
            'price_median': stats_df['price'].median(),
            'price_mean': stats_df['price'].mean(),
            'price_std': stats_df['price'].std(),
            
            # Sqft stats
            'property_sqft_q1': stats_df['property_sqft'].quantile(0.25),
            'property_sqft_q3': stats_df['property_sqft'].quantile(0.75),
            'property_sqft_median': stats_df['property_sqft'].median(),
            'property_sqft_mean': stats_df['property_sqft'].mean(),
            'property_sqft_std': stats_df['property_sqft'].std()
        }
        lookup_rows.append(row)
    
    lookup = pd.DataFrame(lookup_rows)
    stats_time = time.time() - stats_start_time
    
    print(f"\nLookup table created with {len(lookup):,} segments")
    print(f"Statistics computation completed in {stats_time:.2f} seconds ({stats_time/60:.2f} minutes)")
    
    # Display summary
    print("\nSegment size distribution:")
    print(lookup['segment_count'].describe())
    
    print("\nRelaxation level distribution:")
    print(lookup['relaxation_level'].value_counts())
    
    return lookup



def main():
    """
    Main function to build and save lookup table.
    """
    # Setup logging
    log_file = f"../logs/build_lookup_table_{version[:-3]}.txt"
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Redirect stdout to both console and file
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_file)
    
    script_start_time = time.time()
    print("="*80)
    print("Building Lookup Table for All Housing Types")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load data
    data_path = f"../datasets/{version}/combined_listings_{version}.parquet"
    
    print(f"\nLoading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} listings")
    
    # Preprocessing
    print("\nPreprocessing...")
    
    df['offering_type_name'] = df['offering_type_id'].astype(str).replace(OFFERING_TYPE_ID_TO_NAME)
    df['housing_type_name'] = df['housing_type_id'].astype(str).replace(HOUSING_TYPE_ID_MAP)
    df['property_price_type_name'] = df['property_price_type_id'].astype(str).replace(PRICE_TYPE_MAP)
    
    # Filter valid listings
    df = df[(df['price'] > 0) & (df['property_sqft'] > 0)].copy()
    df['price_to_sqft'] = df['price'] / df['property_sqft']
    
    print(f"After filtering (price > 0 and sqft > 0): {len(df):,} listings")
    
    # Filter for online and recent listings
    from datetime import timedelta
    
    n_days_ago = 30
    thirty_days_ago = datetime.now() - timedelta(days=n_days_ago)
    
    online = df['end_year'] == 9999
    recent = df['end_time'] >= thirty_days_ago
    
    # Separate Top Residential and Rest of Housing Types
    is_top_residential_type = df['housing_type_name'].isin(TOP_RESIDENTIAL_TYPES)
    is_rest_housing_type = df['housing_type_name'].isin(REST_OF_HOUSING_TYPES)
    
    df_top_residential = df[(online | recent) & is_top_residential_type].copy()
    df_rest_housing = df[(online | recent) & is_rest_housing_type].copy()

    # Save training set (concatenation of both filtered datasets)
    training_set = pd.concat([df_top_residential, df_rest_housing], ignore_index=True)
    training_set_path = f"../datasets/{version}/training_set_{version}.parquet"
    training_set.to_parquet(training_set_path, index=False)
    print(f"\nSaved training set with {len(training_set):,} listings to: {training_set_path}")
    print(f"\nOnline/recent top residential listings: {len(df_top_residential):,}")
    print(f"Online/recent rest of housing types: {len(df_rest_housing):,}")
    
    print(f"\nTop Residential housing type distribution:")
    print(df_top_residential['housing_type_name'].value_counts())
    
    print(f"\nRest of housing type distribution:")
    print(df_rest_housing['housing_type_name'].value_counts())
    
    # Build and save location tree lookup
    print("\n" + "="*80)
    print("Building location tree lookup...")
    print("="*80)
    loc_cols = ['location_id', 'location_lvl_0_id', 'location_lvl_0_name', 
                'location_lvl_1_id', 'location_lvl_1_name', 'location_lvl_2_id', 
                'location_lvl_2_name', 'location_lvl_3_id', 'location_lvl_3_name', 
                'location_lvl_4_id', 'location_lvl_4_name', 'location_lvl_5_id', 
                'location_lvl_5_name', 'location_lvl_6_id', 'location_lvl_6_name', 
                'location_lvl_7_id', 'location_lvl_7_name']
    
    # Get unique location combinations from the full dataset
    location_tree_lookup = df[loc_cols].drop_duplicates(subset=loc_cols)
    location_tree_path = f"../lookup_tables/location_tree_lookup.parquet"
    location_tree_lookup.to_parquet(location_tree_path, index=False)
    print(f"Saved location tree lookup with {len(location_tree_lookup):,} unique locations to: {location_tree_path}")
    
    # Build lookup tables for both categories
    print("\n" + "="*80)
    print("Building lookup table for Top Residential Types...")
    print("="*80)
    lookup_table_residential = build_lookup_table(
        df_top_residential, 
        MIN_SEGMENT_SIZE,
        TOP_RESIDENTIAL_TYPES
    )
    
    print("\n" + "="*80)
    print("Building lookup table for Rest of Housing Types...")
    print("="*80)
    lookup_table_rest = build_lookup_table(
        df_rest_housing, 
        MIN_SEGMENT_SIZE_REST,
        REST_OF_HOUSING_TYPES
    )
    
    # Combine lookup tables
    print("\n" + "="*80)
    print("Combining lookup tables...")
    lookup_table = pd.concat([lookup_table_residential, lookup_table_rest], ignore_index=True)
    print(f"Combined lookup table: {len(lookup_table):,} segments")
    print("="*80)
    
    # Save lookup table (without feedback applied)
    # To apply CX feedback, run: python src/apply_feedback_to_lookup.py <input> <output>
    output_path = f"../lookup_tables/segment_lookup_table_{version}.parquet"
    print(f"\nSaving lookup table to: {output_path}")
    lookup_table.to_parquet(output_path, index=False)
    
    # Also save as CSV for easy inspection
    csv_path = f"../lookup_tables/segment_lookup_table_{version}.csv"
    print(f"Saving lookup table (CSV) to: {csv_path}")
    lookup_table.to_csv(csv_path, index=False)
    
    script_time = time.time() - script_start_time
    
    print("\n" + "="*80)
    print("Lookup table creation complete!")
    print(f"  Top Residential segments: {len(lookup_table_residential):,}")
    print(f"  Rest of Housing segments: {len(lookup_table_rest):,}")
    print(f"  Total segments: {len(lookup_table):,}")
    print(f"\nTotal execution time: {script_time:.2f} seconds ({script_time/60:.2f} minutes)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Display sample
    print("\nSample lookup table entries:")
    print(lookup_table.head(10).to_string())
    
    # Close log file
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
