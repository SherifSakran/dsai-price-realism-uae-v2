"""
Inference script for outlier detection using lookup table.

This script takes a listings file and a lookup table, then flags outliers
based on deviation from segment medians.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent))
from configs.dataset_config import (
    DATA_DATE_SUFFIX, LOOKUP_TABLE_VERSION, WITH_FEEDBACK
)
from configs.outlier_config import (
    TOP_RESIDENTIAL_TYPES,
    REST_OF_HOUSING_TYPES,
    SEGMENT_SIZE_THRESHOLD,
    PTS_MULTIPLIER,
    PRICE_MULTIPLIER,
    SQFT_MULTIPLIER,
    PTS_LARGER_MULTIPLIER,
    PRICE_LARGER_MULTIPLIER,
    SQFT_LARGER_MULTIPLIER,
    APPLY_LOCATION_ROLLUP,
    LOCATION_MULTIPLIERS,
    OFFERING_TYPE_ID_TO_NAME,
    HOUSING_TYPE_ID_MAP,
    PRICE_TYPE_MAP,
    FEEDBACK_MULTIPLIER_BOOST
)
from src.segment_outlier_detection import create_segment_key

version = DATA_DATE_SUFFIX
lookup_version = LOOKUP_TABLE_VERSION
feedback = ""
if WITH_FEEDBACK:
    lookup_version += '_with_feedback'
    feedback = "_with_feedback"

def assign_segments_from_lookup(df: pd.DataFrame, 
                                lookup_table: pd.DataFrame,
                                location_tree_lookup: pd.DataFrame = None) -> pd.DataFrame:
    """
    Assign segments to listings using the lookup table.
    
    Implements two-stage relaxation:
    1. Bedroom relaxation (for Top Residential Types): try with bedrooms, then without
    2. Location rollup (for Rest of Housing Types): try location_id, then roll up through
       location_lvl_3_id -> location_lvl_2_id -> location_lvl_1_id -> location_lvl_0_id
    
    Args:
        df: DataFrame with listings (must have location_id column)
        lookup_table: Lookup table with segment definitions and stats
        location_tree_lookup: Location tree lookup for getting location hierarchy levels
    
    Returns:
        DataFrame with segment assignments and stats from lookup table
    """
    df = df.copy()
    
    # Load location tree lookup if not provided
    if location_tree_lookup is None:
        location_tree_path = f"../lookup_tables/location_tree_lookup.parquet"
        try:
            location_tree_lookup = pd.read_parquet(location_tree_path)
            print(f"Loaded location tree lookup with {len(location_tree_lookup):,} unique locations")
        except FileNotFoundError:
            print(f"Warning: Location tree lookup not found at {location_tree_path}")
            print("Location rollup for Rest of Housing Types will be limited to location_id only")
            location_tree_lookup = pd.DataFrame()
    
    # Merge location hierarchy levels into the dataframe
    if not location_tree_lookup.empty and 'location_id' in df.columns:
        # Get location columns that don't already exist in df
        loc_cols = [col for col in location_tree_lookup.columns if col not in df.columns or col == 'location_id']
        df = df.merge(
            location_tree_lookup[loc_cols],
            on='location_id',
            how='left'
        )
    
    # Initialize columns
    df['segment_key'] = None
    df['unseen'] = True
    df['relaxation_level'] = None
    
    # Create lookup dictionary for faster matching
    lookup_dict = lookup_table.set_index('segment_key').to_dict('index')
    
    # Determine housing type category for each listing
    is_top_residential = df['housing_type_name'].isin(TOP_RESIDENTIAL_TYPES)
    is_rest_housing = df['housing_type_name'].isin(REST_OF_HOUSING_TYPES)
    
    # ── Stage 1: Bedroom Relaxation (for Top Residential Types) ────────────
    # Try with bedrooms first, then without
    for include_bedrooms in [True, False]:
        # Get unassigned Top Residential listings
        mask = df['unseen'] & is_top_residential
        if not mask.any():
            break
        
        # Create segment keys at location_id level
        df.loc[mask, 'temp_segment'] = df[mask].apply(
            lambda row: create_segment_key(row, include_bedrooms, 'location_id'),
            axis=1
        )
        
        # Match with lookup table
        for idx in df[mask].index:
            segment_key = df.loc[idx, 'temp_segment']
            if segment_key in lookup_dict:
                df.loc[idx, 'segment_key'] = segment_key
                df.loc[idx, 'unseen'] = False
                
                # Copy all stats from lookup table
                for col, val in lookup_dict[segment_key].items():
                    df.loc[idx, col] = val
    
    # ── Stage 2: Location Rollup (for Rest of Housing Types) ───────────────
    # Location hierarchy to try (from most specific to most general)
    if APPLY_LOCATION_ROLLUP:
        location_levels = ['location_id', 'location_lvl_3_id', 'location_lvl_2_id', 
                          'location_lvl_1_id', 'location_lvl_0_id']
    else:
        # When rollup is disabled, only try location_id
        location_levels = ['location_id']
    
    for location_level in location_levels:
        # Get unassigned Rest of Housing listings
        mask = df['unseen'] & is_rest_housing
        if not mask.any():
            break
        
        # Skip if location level column doesn't exist
        if location_level not in df.columns:
            continue
        
        # Create segment keys at this location level (no bedrooms for Rest of Housing)
        df.loc[mask, 'temp_segment'] = df[mask].apply(
            lambda row: create_segment_key(row, include_bedrooms=False, location_level=location_level),
            axis=1
        )
        
        # Match with lookup table
        for idx in df[mask].index:
            segment_key = df.loc[idx, 'temp_segment']
            if segment_key in lookup_dict:
                df.loc[idx, 'segment_key'] = segment_key
                df.loc[idx, 'unseen'] = False
                
                # Copy all stats from lookup table
                for col, val in lookup_dict[segment_key].items():
                    df.loc[idx, col] = val
    
    # Drop temporary column
    if 'temp_segment' in df.columns:
        df = df.drop(columns=['temp_segment'])
    
    # Mark unseen listings
    df.loc[df['unseen'], 'relaxation_level'] = 'unseen'
    
    return df


def compute_bounds_and_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute bounds and outlier flags based on deviation from median.
    Matches the notebook logic exactly: uses two separate threshold checks (3x and 4x)
    and combines them with OR.
    
    Args:
        df: DataFrame with segment assignments and stats
    
    Returns:
        DataFrame with bounds and outlier flags
    """
    df = df.copy()
    
    # Filter out no_segment listings (matching notebook)
    df = df[df['relaxation_level'] != 'unseen'].copy()
    
    # Determine which attributes are available
    has_price = 'price' in df.columns
    has_sqft = 'property_sqft' in df.columns
    has_pts = 'price_to_sqft' in df.columns
    
    # Strip _cx suffix to get base relaxation level for multiplier assignment
    # Feedback-adjusted segments have relaxation_level like 'with_bedrooms_cx'
    df['base_relaxation_level'] = df['relaxation_level'].str.replace('_cx$', '', regex=True)
    is_cx = df['relaxation_level'].str.endswith('_cx', na=False)
    
    # Define segment categories (matching notebook)
    large_segments = df['segment_count'] >= SEGMENT_SIZE_THRESHOLD
    normal_segments = df['segment_count'] < SEGMENT_SIZE_THRESHOLD
    
    # Determine multiplier based on base relaxation_level
    # For Top Residential Types: use 3x or 4x based on segment size and bedrooms
    # For Rest of Housing Types: use location-based multipliers (3x to 7x)
    df['pts_multiplier'] = PTS_MULTIPLIER  # Default: 3
    df['price_multiplier'] = PRICE_MULTIPLIER  # Default: 3
    df['sqft_multiplier'] = SQFT_MULTIPLIER  # Default: 3
    
    # Apply location-based multipliers for Rest of Housing Types
    for location_level, multiplier_value in LOCATION_MULTIPLIERS.items():
        location_mask = df['base_relaxation_level'] == location_level
        df.loc[location_mask, 'pts_multiplier'] = multiplier_value
        df.loc[location_mask, 'price_multiplier'] = multiplier_value
        df.loc[location_mask, 'sqft_multiplier'] = multiplier_value
    
    # For Top Residential Types: override with larger multiplier for large segments or without_bedrooms
    # (matching the notebook logic: use 4x when EITHER condition is met)
    residential_mask = df['base_relaxation_level'].isin(['with_bedrooms', 'without_bedrooms'])
    use_larger = residential_mask & (large_segments | (df['base_relaxation_level'] == 'without_bedrooms'))
    df.loc[use_larger, 'pts_multiplier'] = PTS_LARGER_MULTIPLIER
    df.loc[use_larger, 'price_multiplier'] = PRICE_LARGER_MULTIPLIER
    df.loc[use_larger, 'sqft_multiplier'] = SQFT_LARGER_MULTIPLIER
    
    # Apply FEEDBACK_MULTIPLIER_BOOST to _cx segments (feedback-adjusted)
    df.loc[is_cx, 'pts_multiplier'] *= FEEDBACK_MULTIPLIER_BOOST
    df.loc[is_cx, 'price_multiplier'] *= FEEDBACK_MULTIPLIER_BOOST
    df.loc[is_cx, 'sqft_multiplier'] *= FEEDBACK_MULTIPLIER_BOOST
    
    # Price to sqft outliers (only if both price and sqft exist)
    if has_pts:
        # Use the assigned multiplier for each listing
        high_pts = df['price_to_sqft'] > df['pts_multiplier'] * df['price_to_sqft_median']
        low_pts = df['price_to_sqft'] < (1.0 / df['pts_multiplier']) * df['price_to_sqft_median']
        
        df['is_pts_outlier'] = high_pts | low_pts
        
        df['pts_outlier_type'] = 'not_outlier'
        df.loc[low_pts, 'pts_outlier_type'] = 'low'
        df.loc[high_pts, 'pts_outlier_type'] = 'high'
        
        # Calculate deviation ratio: how many times the value deviates from median
        # For high outliers: value / median (e.g., 10x median)
        # For low outliers: median / value (e.g., 5x below median)
        df['pts_deviation_ratio'] = df['price_to_sqft'] / df['price_to_sqft_median']
        df.loc[low_pts, 'pts_deviation_ratio'] = df.loc[low_pts, 'price_to_sqft_median'] / df.loc[low_pts, 'price_to_sqft']
    
    # Price outliers (only if price exists)
    if has_price:
        # Use the assigned multiplier for each listing
        high_price = df['price'] > df['price_multiplier'] * df['price_median']
        low_price = df['price'] < (1.0 / df['price_multiplier']) * df['price_median']
        
        df['is_price_outlier'] = high_price | low_price
        
        df['price_outlier_type'] = 'not_outlier'
        df.loc[low_price, 'price_outlier_type'] = 'low'
        df.loc[high_price, 'price_outlier_type'] = 'high'
        
        # Calculate deviation ratio
        df['price_deviation_ratio'] = df['price'] / df['price_median']
        df.loc[low_price, 'price_deviation_ratio'] = df.loc[low_price, 'price_median'] / df.loc[low_price, 'price']
    
    # Sqft outliers (only if sqft exists)
    if has_sqft:
        # Use the assigned multiplier for each listing
        high_sqft = df['property_sqft'] > df['sqft_multiplier'] * df['property_sqft_median']
        low_sqft = df['property_sqft'] < (1.0 / df['sqft_multiplier']) * df['property_sqft_median']
        
        df['is_sqft_outlier'] = high_sqft | low_sqft
        
        df['sqft_outlier_type'] = 'not_outlier'
        df.loc[low_sqft, 'sqft_outlier_type'] = 'low'
        df.loc[high_sqft, 'sqft_outlier_type'] = 'high'
        
        # Calculate deviation ratio
        df['sqft_deviation_ratio'] = df['property_sqft'] / df['property_sqft_median']
        df.loc[low_sqft, 'sqft_deviation_ratio'] = df.loc[low_sqft, 'property_sqft_median'] / df.loc[low_sqft, 'property_sqft']
    
    # Drop helper column
    if 'base_relaxation_level' in df.columns:
        df = df.drop(columns=['base_relaxation_level'])
    
    return df


def run_inference(df: pd.DataFrame, 
                 lookup_table: pd.DataFrame,
                 filter_online: bool = True) -> pd.DataFrame:
    """
    Run outlier detection inference on listings.
    
    Args:
        df: DataFrame with listings
        lookup_table: Lookup table with segment definitions and stats
        filter_online: Whether to filter for online listings only
    
    Returns:
        DataFrame with outlier flags and segment information
    """
    # Validate required columns for segment assignment
    required_cols = [
        'location_id', 'housing_type_name', 'offering_type_name',
        'property_price_type_name', 'bedrooms'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = df.copy()
    
    # Convert to float to handle Decimal types (if columns exist)
    if 'price' in df.columns and 'property_sqft' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['property_sqft'] = pd.to_numeric(df['property_sqft'], errors='coerce')
        # Compute price_to_sqft
        df['price_to_sqft'] = df['price'] / df['property_sqft']
    elif 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    elif 'property_sqft' in df.columns:
        df['property_sqft'] = pd.to_numeric(df['property_sqft'], errors='coerce')
    
    # Filter for online listings if requested
    if filter_online:
        online_mask = df['end_year'] == 9999
        df = df[online_mask].copy()
        print(f"Filtered to online listings: {len(df):,}")
    
    # Assign segments from lookup table
    print("Assigning segments from lookup table...")
    df = assign_segments_from_lookup(df, lookup_table)
    
    # Summary of segment assignment
    assigned = (~df['unseen']).sum()
    unseen = df['unseen'].sum()
    
    print(f"\nSegment assignment results:")
    print(f"  Assigned to segments: {assigned:,} ({assigned/len(df)*100:.1f}%)")
    print(f"  Unseen segments: {unseen:,} ({unseen/len(df)*100:.1f}%)")
    
    # Add is_realistic column for manual review (empty by default)
    df.insert(1, 'is_realistic', pd.NA)
    df.insert(2, 'reason', pd.NA)

    # Compute bounds and flags
    print("\nComputing bounds and outlier flags...")
    df = compute_bounds_and_flags(df)
    
    # Summary statistics (only for attributes that were computed)
    if assigned > 0:
        print(f"\nOutlier detection results:")
        
        outlier_flags = []
        
        if 'is_pts_outlier' in df.columns:
            pts_outliers = df['is_pts_outlier'].sum()
            print(f"  Price-to-sqft outliers: {pts_outliers:,} ({pts_outliers/assigned*100:.1f}%)")
            outlier_flags.append(df['is_pts_outlier'].fillna(False))
        
        if 'is_price_outlier' in df.columns:
            price_outliers = df['is_price_outlier'].sum()
            print(f"  Price outliers: {price_outliers:,} ({price_outliers/assigned*100:.1f}%)")
            outlier_flags.append(df['is_price_outlier'].fillna(False))
        
        if 'is_sqft_outlier' in df.columns:
            sqft_outliers = df['is_sqft_outlier'].sum()
            print(f"  Sqft outliers: {sqft_outliers:,} ({sqft_outliers/assigned*100:.1f}%)")
            outlier_flags.append(df['is_sqft_outlier'].fillna(False))
        
        # Combined outlier flag (only if at least one attribute was flagged)
        if outlier_flags:
            df['is_any_outlier'] = outlier_flags[0]
            for flag in outlier_flags[1:]:
                df['is_any_outlier'] = df['is_any_outlier'] | flag
            
            any_outliers = df['is_any_outlier'].sum()
            print(f"  At least one outlier: {any_outliers:,} ({any_outliers/assigned*100:.1f}%)")
            
            # Rank outliers by deviation severity (most obvious to least obvious)
            print("\nRanking outliers by deviation severity...")
            df = rank_outliers_by_deviation(df)
    
    return df


def rank_outliers_by_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank outliers by deviation severity from most obvious to least obvious.
    
    Accounts for relaxation level by normalizing deviation ratios by the multiplier threshold.
    A 2x deviation with multiplier=3 (tight segment) is more severe than 2x deviation with multiplier=6 (relaxed segment).
    
    Normalized score = (deviation_ratio - 1) / (multiplier - 1)
    - Score > 1.0 means exceeds the threshold
    - Higher score = more obvious outlier
    
    Ranking priority:
    1. Price-to-sqft normalized score (primary)
    2. Price normalized score (secondary)
    3. Sqft normalized score (tertiary)
    
    Args:
        df: DataFrame with outlier flags, deviation ratios, and multipliers
    
    Returns:
        DataFrame with outlier_rank column (1 = most obvious outlier)
    """
    # Initialize deviation ratios to 1.0 (no deviation) for non-outliers
    if 'pts_deviation_ratio' not in df.columns:
        df['pts_deviation_ratio'] = 1.0
    if 'price_deviation_ratio' not in df.columns:
        df['price_deviation_ratio'] = 1.0
    if 'sqft_deviation_ratio' not in df.columns:
        df['sqft_deviation_ratio'] = 1.0
    
    # Fill NaN values with 1.0 (no deviation)
    df['pts_deviation_ratio'] = df['pts_deviation_ratio'].fillna(1.0)
    df['price_deviation_ratio'] = df['price_deviation_ratio'].fillna(1.0)
    df['sqft_deviation_ratio'] = df['sqft_deviation_ratio'].fillna(1.0)
    
    # Ensure multiplier columns exist (should be set by compute_bounds_and_flags)
    if 'pts_multiplier' not in df.columns:
        df['pts_multiplier'] = PTS_MULTIPLIER
    if 'price_multiplier' not in df.columns:
        df['price_multiplier'] = PRICE_MULTIPLIER
    if 'sqft_multiplier' not in df.columns:
        df['sqft_multiplier'] = SQFT_MULTIPLIER
    
    # Calculate normalized deviation scores
    # Formula: (deviation_ratio - 1) / (multiplier - 1)
    # This normalizes by how much the threshold allows
    df['pts_normalized_score'] = (df['pts_deviation_ratio'] - 1) / (df['pts_multiplier'] - 1)
    df['price_normalized_score'] = (df['price_deviation_ratio'] - 1) / (df['price_multiplier'] - 1)
    df['sqft_normalized_score'] = (df['sqft_deviation_ratio'] - 1) / (df['sqft_multiplier'] - 1)
    
    # Create a copy for ranking (only outliers)
    if 'is_any_outlier' in df.columns:
        outliers_mask = df['is_any_outlier'] == True
        
        if outliers_mask.any():
            # Sort outliers by normalized scores (descending order - highest score first)
            # Priority: pts_normalized_score > price_normalized_score > sqft_normalized_score
            df_outliers = df[outliers_mask].copy()
            df_outliers = df_outliers.sort_values(
                by=['pts_normalized_score', 'price_normalized_score', 'sqft_normalized_score'],
                ascending=[False, False, False]
            )
            
            # Assign ranks (1 = most obvious outlier)
            df_outliers['outlier_rank'] = range(1, len(df_outliers) + 1)
            
            # Merge ranks back to original dataframe
            df = df.merge(
                df_outliers[['property_listing_id', 'outlier_rank']],
                on='property_listing_id',
                how='left'
            )
            
            print(f"  Ranked {len(df_outliers):,} outliers by normalized deviation severity")
            print(f"  Top 5 most obvious outliers:")
            top_5 = df_outliers.head(5)[['property_listing_id', 'housing_type_name', 'relaxation_level',
                                          'pts_deviation_ratio', 'pts_multiplier', 'pts_normalized_score', 
                                          'outlier_rank']]
            for _, row in top_5.iterrows():
                print(f"    Rank {int(row['outlier_rank'])}: {row['housing_type_name']} ({row['relaxation_level']}) - "
                      f"PTS: {row['pts_deviation_ratio']:.1f}x (threshold: {row['pts_multiplier']:.0f}x, "
                      f"score: {row['pts_normalized_score']:.2f})")
    
    return df


def main():
    """
    Main function for inference.
    """
    print("=" * 80)
    print("Outlier Detection Inference")
    print("=" * 80)
    
    # Load lookup table
    lookup_path = f"../lookup_tables/segment_lookup_table_{lookup_version}.parquet"
    
    print(f"\nLoading lookup table from: {lookup_path}")
    lookup_table = pd.read_parquet(lookup_path)
    print(f"Loaded lookup table with {len(lookup_table):,} segments")
    
    # Filter out deprecated segments (kept for reference only)
    if 'deprecated' in lookup_table.columns:
        deprecated_count = lookup_table['deprecated'].sum()
        lookup_table = lookup_table[~lookup_table['deprecated']].copy()
        print(f"Filtered out {deprecated_count} deprecated segments (kept for reference only)")
        print(f"Active segments: {len(lookup_table):,}")
    
    # Load listings data
    data_path = f"../datasets/{version}/combined_listings_{version}.parquet"
    
    print(f"\nLoading listings from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} listings")
    
    # Preprocessing
    print("\nPreprocessing...")
    
    df['offering_type_name'] = df['offering_type_id'].astype(str).replace(OFFERING_TYPE_ID_TO_NAME)
    df['housing_type_name'] = df['housing_type_id'].astype(str).replace(HOUSING_TYPE_ID_MAP)
    df['property_price_type_name'] = df['property_price_type_id'].astype(str).replace(PRICE_TYPE_MAP)
    
    # Filter valid listings
    # df = df[(df['price'] > 0) & (df['property_sqft'] > 0)].copy()
    # print(f"After filtering (price > 0 and sqft > 0): {len(df):,} listings")

    # Filter for online and recent listings (both categories)
    from datetime import datetime, timedelta
    
    n_days_ago = 30
    thirty_days_ago = datetime.now() - timedelta(days=n_days_ago)
    
    online = df['end_year'] == 9999
    recent = df['end_time'] >= thirty_days_ago
    is_top_residential_type = df['housing_type_name'].isin(TOP_RESIDENTIAL_TYPES)
    is_rest_housing_type = df['housing_type_name'].isin(REST_OF_HOUSING_TYPES)
    
    df_filtered = df[(online | recent) & (is_top_residential_type | is_rest_housing_type)].copy()
    print(f"Online/recent top residential listings: {(df_filtered['housing_type_name'].isin(TOP_RESIDENTIAL_TYPES)).sum():,}")
    print(f"Online/recent rest of housing types: {(df_filtered['housing_type_name'].isin(REST_OF_HOUSING_TYPES)).sum():,}")
    print(f"Total online/recent listings: {len(df_filtered):,}")
    
    # Run inference on all (online | recent) listings
    print("\n" + "=" * 80)
    print("Running inference on all housing types...")
    df_flagged = run_inference(df_filtered, lookup_table, filter_online=True)
    print("=" * 80)
    
    # Save results
    output_path = f"../datasets/{version}/inference_flagged_listings_{version}{feedback}.parquet"
    print(f"\nSaving flagged listings to: {output_path}")
    df_flagged.to_parquet(output_path, index=False)
    
    # Save outliers only (filter to online only, matching notebook)
    online_flagged = df_flagged['end_year'] == 9999
    outliers = df_flagged[online_flagged & df_flagged['is_any_outlier']].copy()

    # Add URL column
    outliers['url'] = 'http://propertyfinder.ae/to/' + outliers['property_listing_id'].astype(str)

    # Sort outliers by rank (most obvious first)
    if 'outlier_rank' in outliers.columns:
        outliers = outliers.sort_values('outlier_rank')
        print(f"Outliers sorted by rank (1 = most obvious)")

    outliers_path = f"../datasets/{version}/inference_outliers_only_{version}{feedback}.parquet"
    print(f"Saving online outliers only to: {outliers_path}")
    print(f"Total online outliers: {len(outliers):,}")
    outliers.to_parquet(outliers_path, index=False)
    
    # Save as CSV for inspection
    csv_path = f"../datasets/{version}/inference_outliers_only_{version}{feedback}.csv"
    print(f"Saving outliers (CSV) to: {csv_path}")
    
    # Select key columns for CSV
    csv_cols = [
        'outlier_rank',  # Add rank as first column
        'url', 'is_realistic', 'property_listing_id', 'location_lvl_0_name', 'location_id',
        'housing_type_name', 'offering_type_name','property_price_type_name', 'bedrooms', 'price', 'property_sqft', 'price_to_sqft',
        'segment_key', 'segment_count', 'relaxation_level',
        'price_median', 'property_sqft_median', 'price_to_sqft_median',
        'pts_deviation_ratio', 'price_deviation_ratio', 'sqft_deviation_ratio',  # Raw deviation ratios
        'pts_normalized_score', 'price_normalized_score', 'sqft_normalized_score',  # Normalized scores (account for relaxation)
        'is_pts_outlier', 'pts_outlier_type',
        'is_price_outlier', 'price_outlier_type',
        'is_sqft_outlier', 'sqft_outlier_type',
        'pts_multiplier', 'price_multiplier', 'sqft_multiplier',
        'property_permit_number', 'listing_short_term_flag', 'complaince_type', 'client_id', 'salesforce_account_no'
    ]
    
    # Only include columns that exist
    csv_cols = [col for col in csv_cols if col in outliers.columns]
    outliers[csv_cols].to_csv(csv_path, index=False)
    
    print("\n" + "=" * 80)
    print("Inference complete!")
    print(f"  Processed listings: {len(df_flagged):,}")
    if 'is_any_outlier' in df_flagged.columns:
        top_res_outliers = df_flagged[df_flagged['housing_type_name'].isin(TOP_RESIDENTIAL_TYPES) & df_flagged['is_any_outlier']]
        rest_outliers = df_flagged[df_flagged['housing_type_name'].isin(REST_OF_HOUSING_TYPES) & df_flagged['is_any_outlier']]
        print(f"  Top Residential outliers: {len(top_res_outliers):,}")
        print(f"  Rest of Housing outliers: {len(rest_outliers):,}")
    print("=" * 80)
    
    # Display sample outliers
    if len(outliers) > 0:
        print("\nSample outliers:")
        display_cols = [
            'url', 'housing_type_name', 'bedrooms', 'price', 'property_sqft', 'price_to_sqft',
            'is_pts_outlier', 'is_price_outlier', 'is_sqft_outlier'
        ]
        display_cols = [col for col in display_cols if col in outliers.columns]
        print(outliers[display_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
