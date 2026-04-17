import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from configs.outlier_config import APPLY_LOCATION_ROLLUP


TOP_RESIDENTIAL_TYPES = [
    'Apartment', 'Villa', 'Townhouse', 
    'Hotel & Hotel Apartment', 'Penthouse', 'Duplex'
]

REST_OF_HOUSING_TYPES = [
    'Office Space', 'Land', 'Warehouse', 'Shop', 'Retail',
    'Whole Building', 'Business Centre', 'Labor Camp', 'Show Room',
    'Full Floor', 'Factory', 'Compound', 'Staff Accommodation',
    'Co-working space', 'Half Floor', 'Farm', 'Bulk Rent Unit',
    'Bulk Sale Unit', 'Bungalow'
]

MIN_SEGMENT_SIZE = {
    'Apartment': 5,
    'Villa': 5,
    'Townhouse': 5,
    'Hotel & Hotel Apartment': 5,
    'Penthouse': 5,
    'Duplex': 5
}

MIN_SEGMENT_SIZE_REST = {
    'Office Space': 10,
    'Land': 10,
    'Warehouse': 10,
    'Shop': 15,
    'Retail': 15,
    'Whole Building': 15,
    'Business Centre': 15,
    'Labor Camp': 20,
    'Show Room': 20,
    'Full Floor': 20,
    'Compound': 20,
    'Factory': 20,
    'Staff Accommodation': 20,
    'Co-working space': 25,
    'Bulk Rent Unit': 25,
    'Half Floor': 25,
    'Bulk Sale Unit': 30,
    'Farm': 30,
    'Bungalow': 40
}


def create_segment_key(row: pd.Series, include_bedrooms: bool = True, 
                       location_level: str = 'location_id') -> str:
    """
    Create a segment key for a listing based on segmentation rules.
    
    Args:
        row: DataFrame row containing listing attributes
        include_bedrooms: Whether to include bedrooms in the segment
        location_level: Which location level to use (location_id, location_lvl_3_id, etc.)
    
    Returns:
        String key representing the segment
    """
    parts = [
        str(row.get(location_level, '')),
        str(row.get('housing_type_name', '')),
        str(row.get('offering_type_name', '')),
        str(row.get('property_price_type_name', ''))
    ]
    
    if include_bedrooms:
        parts.append(str(row.get('bedrooms', '')))
    
    return '|'.join(parts)


def assign_segments_with_relaxation(df: pd.DataFrame, 
                                    min_segment_sizes: Dict[str, int] = None) -> pd.DataFrame:
    """
    Assign each listing to a segment, applying relaxation when needed.
    
    Relaxation hierarchy for Top Residential Types:
    1. Full segment: location_id + housing_type + offering_type + price_type + bedrooms
    2. Relax bedrooms: location_id + housing_type + offering_type + price_type
    
    Relaxation hierarchy for Rest of Housing Types:
    1. location_id + housing_type + offering_type + price_type
    2. location_lvl_3_id + housing_type + offering_type + price_type
    3. location_lvl_2_id + housing_type + offering_type + price_type
    4. location_lvl_1_id + housing_type + offering_type + price_type
    5. location_lvl_0_id + housing_type + offering_type + price_type
    
    Args:
        df: DataFrame with listings
        min_segment_sizes: Dictionary mapping housing types to minimum segment sizes
    
    Returns:
        DataFrame with segment assignments and metadata
    """
    if min_segment_sizes is None:
        min_segment_sizes = {**MIN_SEGMENT_SIZE, **MIN_SEGMENT_SIZE_REST}
    
    df = df.copy()
    
    # Initialize segment columns
    df['segment_key'] = None
    df['segment_size'] = 0
    df['relaxation_level'] = None
    
    # Separate Top Residential and Rest of Housing Types
    is_top_residential = df['housing_type_name'].isin(TOP_RESIDENTIAL_TYPES)
    is_rest_housing = df['housing_type_name'].isin(REST_OF_HOUSING_TYPES)
    
    # Process Top Residential Types (bedroom-based relaxation)
    if is_top_residential.any():
        df_residential = df[is_top_residential].copy()
        df_residential = _assign_segments_residential(df_residential, min_segment_sizes)
        df.loc[is_top_residential, df_residential.columns] = df_residential
    
    # Process Rest of Housing Types (location-based relaxation)
    if is_rest_housing.any():
        df_rest = df[is_rest_housing].copy()
        df_rest = _assign_segments_rest(df_rest, min_segment_sizes)
        df.loc[is_rest_housing, df_rest.columns] = df_rest
    
    # Mark listings that couldn't be assigned to any segment
    df.loc[df['segment_key'].isna(), 'relaxation_level'] = 'no_segment'
    
    return df


def _assign_segments_residential(df: pd.DataFrame, 
                                 min_segment_sizes: Dict[str, int]) -> pd.DataFrame:
    """
    Assign segments for Top Residential Types using bedroom-based relaxation.
    
    Two-pass approach:
    1. Assign with_bedrooms segments (only for bedroom groups >= min_size)
    2. For unassigned listings, check if without_bedrooms segment exists using ALL listings
       (not just unassigned ones) to ensure proper fallback statistics
    
    Args:
        df: DataFrame with residential listings
        min_segment_sizes: Dictionary mapping housing types to minimum segment sizes
    
    Returns:
        DataFrame with segment assignments
    """
    # PASS 1: Assign with_bedrooms segments
    df['temp_segment_with_bed'] = df.apply(
        lambda row: create_segment_key(row, include_bedrooms=True, location_level='location_id'),
        axis=1
    )
    
    # Count segment sizes for with_bedrooms
    segment_counts_with_bed = df.groupby('temp_segment_with_bed').size()
    
    # Assign segments that meet minimum size requirement
    for segment_key, count in segment_counts_with_bed.items():
        # Get housing type from segment
        parts = segment_key.split('|')
        housing_type = parts[1] if len(parts) > 1 else None
        
        # Determine minimum size for this housing type
        min_size = min_segment_sizes.get(housing_type, 5)
        
        if count >= min_size:
            segment_mask = df['temp_segment_with_bed'] == segment_key
            df.loc[segment_mask, 'segment_key'] = segment_key
            df.loc[segment_mask, 'segment_size'] = count
            df.loc[segment_mask, 'relaxation_level'] = "with_bedrooms"
    
    # PASS 2: For unassigned listings, try without_bedrooms using ALL listings
    unassigned_mask = df['segment_key'].isna()
    
    if unassigned_mask.any():
        # Create without_bedrooms segment keys for ALL listings (not just unassigned)
        df['temp_segment_without_bed'] = df.apply(
            lambda row: create_segment_key(row, include_bedrooms=False, location_level='location_id'),
            axis=1
        )
        
        # Count segment sizes using ALL listings
        segment_counts_without_bed = df.groupby('temp_segment_without_bed').size()
        
        # Assign unassigned listings to without_bedrooms segments that meet minimum size
        for segment_key, count in segment_counts_without_bed.items():
            # Get housing type from segment
            parts = segment_key.split('|')
            housing_type = parts[1] if len(parts) > 1 else None
            
            # Determine minimum size for this housing type
            min_size = min_segment_sizes.get(housing_type, 5)
            
            if count >= min_size:
                # Only assign to unassigned listings
                segment_mask = (df['temp_segment_without_bed'] == segment_key) & unassigned_mask
                df.loc[segment_mask, 'segment_key'] = segment_key
                df.loc[segment_mask, 'segment_size'] = count
                df.loc[segment_mask, 'relaxation_level'] = "without_bedrooms"
        
        # Drop temporary column
        df = df.drop(columns=['temp_segment_without_bed'])
    
    # Drop temporary column
    df = df.drop(columns=['temp_segment_with_bed'])
    
    return df


def _assign_segments_rest(df: pd.DataFrame, 
                         min_segment_sizes: Dict[str, int]) -> pd.DataFrame:
    """
    Assign segments for Rest of Housing Types using location-based relaxation.
    
    Handles cases where location_id may equal location_lvl_2_id or location_lvl_1_id
    when deeper levels don't exist (are null).
    
    Args:
        df: DataFrame with non-residential listings
        min_segment_sizes: Dictionary mapping housing types to minimum segment sizes
    
    Returns:
        DataFrame with segment assignments
    """
    # Track which segment keys we've already tried to avoid duplicates
    df['tried_segments'] = [set() for _ in range(len(df))]
    
    # Location hierarchy: from most specific to least specific
    location_levels_full = ['location_id', 'location_lvl_3_id', 'location_lvl_2_id', 
                           'location_lvl_1_id', 'location_lvl_0_id']
    
    for location_level in location_levels_full:
        # Skip if already assigned
        mask = df['segment_key'].isna()
        if not mask.any():
            break
        
        # Skip if location level doesn't exist in dataframe
        if location_level not in df.columns:
            continue
        
        # Skip rows where this location level is null
        valid_location_mask = mask & df[location_level].notna()
        if not valid_location_mask.any():
            continue
        
        # For higher-level locations (not location_id), only process if APPLY_LOCATION_ROLLUP is True
        if location_level != 'location_id' and not APPLY_LOCATION_ROLLUP:
            continue
        
        # Create segment keys for unassigned listings (no bedrooms for Rest types)
        df.loc[valid_location_mask, 'temp_segment'] = df[valid_location_mask].apply(
            lambda row: create_segment_key(row, include_bedrooms=False, location_level=location_level),
            axis=1
        )
        
        # Filter out segment keys we've already tried (handles case where location_id == location_lvl_2_id)
        for idx in df[valid_location_mask].index:
            segment_key = df.loc[idx, 'temp_segment']
            if segment_key in df.loc[idx, 'tried_segments']:
                df.loc[idx, 'temp_segment'] = None
            else:
                df.loc[idx, 'tried_segments'].add(segment_key)
        
        # Update mask to only include rows with new segment keys
        valid_location_mask = valid_location_mask & df['temp_segment'].notna()
        if not valid_location_mask.any():
            continue
        
        # Count segment sizes
        segment_counts = df[valid_location_mask].groupby('temp_segment').size()
        
        # Assign segments that meet minimum size requirement
        for segment_key, count in segment_counts.items():
            # Get housing type from segment
            parts = segment_key.split('|')
            housing_type = parts[1] if len(parts) > 1 else None
            
            # Determine minimum size for this housing type
            min_size = min_segment_sizes.get(housing_type, 10)
            
            if count >= min_size:
                segment_mask = (df['temp_segment'] == segment_key) & valid_location_mask
                df.loc[segment_mask, 'segment_key'] = segment_key
                df.loc[segment_mask, 'segment_size'] = count
                
                # Record relaxation level based on location level
                if location_level == 'location_id':
                    relaxation_desc = 'location_id'
                elif location_level == 'location_lvl_3_id':
                    relaxation_desc = 'location_lvl_3'
                elif location_level == 'location_lvl_2_id':
                    relaxation_desc = 'location_lvl_2'
                elif location_level == 'location_lvl_1_id':
                    relaxation_desc = 'location_lvl_1'
                elif location_level == 'location_lvl_0_id':
                    relaxation_desc = 'location_lvl_0'
                else:
                    relaxation_desc = location_level
                
                df.loc[segment_mask, 'relaxation_level'] = relaxation_desc
    
    # Drop temporary columns
    if 'temp_segment' in df.columns:
        df = df.drop(columns=['temp_segment'])
    if 'tried_segments' in df.columns:
        df = df.drop(columns=['tried_segments'])
    
    return df


def calculate_iqr_bounds(df: pd.DataFrame, 
                        attribute: str = 'price_to_sqft',
                        iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Calculate IQR bounds for each segment.
    
    Args:
        df: DataFrame with segment assignments
        attribute: Column name to calculate bounds for (e.g., 'price_to_sqft')
        iqr_multiplier: Multiplier for IQR to define outlier bounds
    
    Returns:
        DataFrame with IQR bounds per segment
    """
    # Only calculate for listings with valid segments
    valid_df = df[df['segment_key'].notna()].copy()
    
    # Convert attribute to float to handle Decimal types
    valid_df[attribute] = pd.to_numeric(valid_df[attribute], errors='coerce')
    
    # Calculate quartiles and IQR per segment
    bounds = valid_df.groupby('segment_key').agg(
        segment_count=(attribute, 'size'),
        q1=(attribute, lambda x: x.quantile(0.25)),
        q3=(attribute, lambda x: x.quantile(0.75)),
        median=(attribute, 'median'),
        mean=(attribute, 'mean'),
        std=(attribute, 'std')
    ).reset_index()
    
    # Calculate IQR and bounds
    bounds['iqr'] = bounds['q3'] - bounds['q1']
    bounds['lower_bound'] = bounds['q1'] - iqr_multiplier * bounds['iqr']
    bounds['upper_bound'] = bounds['q3'] + iqr_multiplier * bounds['iqr']
    
    # Handle cases where IQR is very small or zero
    # Use a percentage-based approach as fallback
    small_iqr_mask = bounds['iqr'] < (bounds['median'] * 0.1)
    bounds.loc[small_iqr_mask, 'lower_bound'] = bounds.loc[small_iqr_mask, 'median'] * 0.5
    bounds.loc[small_iqr_mask, 'upper_bound'] = bounds.loc[small_iqr_mask, 'median'] * 2.0
    
    # Rename columns to include attribute name
    bounds = bounds.rename(columns={
        'q1': f'{attribute}_q1',
        'q3': f'{attribute}_q3',
        'median': f'{attribute}_median',
        'mean': f'{attribute}_mean',
        'std': f'{attribute}_std',
        'iqr': f'{attribute}_iqr',
        'lower_bound': f'{attribute}_lower_bound',
        'upper_bound': f'{attribute}_upper_bound'
    })
    
    return bounds


def flag_outliers(df: pd.DataFrame, 
                 bounds: pd.DataFrame,
                 attribute: str = 'price_to_sqft') -> pd.DataFrame:
    """
    Flag outliers based on IQR bounds.
    
    Args:
        df: DataFrame with listings and segment assignments
        bounds: DataFrame with IQR bounds per segment
        attribute: Column name to flag outliers for
    
    Returns:
        DataFrame with outlier flags
    """
    df = df.copy()
    
    # Merge bounds
    df = df.merge(
        bounds,
        on='segment_key',
        how='left',
        suffixes=('', '_segment')
    )
    
    # Flag outliers
    lower_col = f'{attribute}_lower_bound'
    upper_col = f'{attribute}_upper_bound'
    
    df[f'is_{attribute}_outlier'] = False
    df[f'{attribute}_outlier_type'] = 'normal'
    
    # Only flag for listings with valid segments
    valid_mask = df['segment_key'].notna()
    
    # Flag low outliers
    low_mask = valid_mask & (df[attribute] < df[lower_col])
    df.loc[low_mask, f'is_{attribute}_outlier'] = True
    df.loc[low_mask, f'{attribute}_outlier_type'] = 'too_low'
    
    # Flag high outliers
    high_mask = valid_mask & (df[attribute] > df[upper_col])
    df.loc[high_mask, f'is_{attribute}_outlier'] = True
    df.loc[high_mask, f'{attribute}_outlier_type'] = 'too_high'
    
    # Calculate deviation from median
    df[f'{attribute}_deviation_from_median'] = (
        (df[attribute] - df[f'{attribute}_median']) / df[f'{attribute}_median']
    ) * 100
    
    return df


def detect_price_to_sqft_outliers(df: pd.DataFrame,
                                  iqr_multiplier: float = 1.5,
                                  min_segment_sizes: Dict[str, int] = None) -> pd.DataFrame:
    """
    Complete pipeline to detect price_to_sqft outliers using segmentation.
    
    Args:
        df: DataFrame with listings (must have required columns)
        iqr_multiplier: Multiplier for IQR bounds (1.5 = standard, higher = more lenient)
        min_segment_sizes: Dictionary mapping housing types to minimum segment sizes
    
    Returns:
        DataFrame with outlier flags and segment information
    """
    # Validate required columns
    required_cols = [
        'location_id', 'housing_type_name', 'offering_type_name',
        'property_price_type_name', 'price_to_sqft'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert price_to_sqft to float to handle Decimal types
    df = df.copy()
    df['price_to_sqft'] = pd.to_numeric(df['price_to_sqft'], errors='coerce')
    
    # Step 1: Assign segments with relaxation
    print("Assigning segments with relaxation...")
    df = assign_segments_with_relaxation(df, min_segment_sizes)
    
    # Step 2: Calculate IQR bounds per segment
    print("Calculating IQR bounds per segment...")
    bounds = calculate_iqr_bounds(df, 'price_to_sqft', iqr_multiplier)
    
    # Step 3: Flag outliers
    print("Flagging outliers...")
    df = flag_outliers(df, bounds, 'price_to_sqft')
    
    # Summary statistics
    total_listings = len(df)
    assigned_segments = df['segment_key'].notna().sum()
    outliers = df['is_price_to_sqft_outlier'].sum()
    
    print(f"\nSummary:")
    print(f"Total listings: {total_listings:,}")
    print(f"Assigned to segments: {assigned_segments:,} ({assigned_segments/total_listings*100:.1f}%)")
    print(f"Outliers detected: {outliers:,} ({outliers/total_listings*100:.1f}%)")
    
    if outliers > 0:
        too_low = (df['price_to_sqft_outlier_type'] == 'too_low').sum()
        too_high = (df['price_to_sqft_outlier_type'] == 'too_high').sum()
        print(f"  - Too low: {too_low:,} ({too_low/outliers*100:.1f}%)")
        print(f"  - Too high: {too_high:,} ({too_high/outliers*100:.1f}%)")
    
    return df


def detect_outliers_multi_attribute(df: pd.DataFrame,
                                    attributes: List[str] = None,
                                    iqr_multiplier: float = 1.5,
                                    min_segment_sizes: Dict[str, int] = None) -> pd.DataFrame:
    """
    Complete pipeline to detect outliers for multiple attributes using segmentation.
    
    Args:
        df: DataFrame with listings (must have required columns)
        attributes: List of attributes to detect outliers for (e.g., ['price', 'property_sqft', 'price_to_sqft'])
        iqr_multiplier: Multiplier for IQR bounds (1.5 = standard, higher = more lenient)
        min_segment_sizes: Dictionary mapping housing types to minimum segment sizes
    
    Returns:
        DataFrame with outlier flags and segment information for all attributes
    """
    if attributes is None:
        attributes = ['price_to_sqft']
    
    # Validate required columns
    required_cols = [
        'location_id', 'housing_type_name', 'offering_type_name',
        'property_price_type_name'
    ] + attributes
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert attributes to float to handle Decimal types
    df = df.copy()
    for attr in attributes:
        df[attr] = pd.to_numeric(df[attr], errors='coerce')
    
    # Step 1: Assign segments with relaxation (only once)
    print("Assigning segments with relaxation...")
    df = assign_segments_with_relaxation(df, min_segment_sizes)
    
    # Step 2-3: Calculate bounds and flag outliers for each attribute
    for attr in attributes:
        print(f"\nProcessing attribute: {attr}")
        print(f"  Calculating IQR bounds...")
        bounds = calculate_iqr_bounds(df, attr, iqr_multiplier)
        
        print(f"  Flagging outliers...")
        df = flag_outliers(df, bounds, attr)
        
        # Summary statistics for this attribute
        outliers = df[f'is_{attr}_outlier'].sum()
        print(f"  Outliers detected: {outliers:,} ({outliers/len(df)*100:.1f}%)")
        
        if outliers > 0:
            too_low = (df[f'{attr}_outlier_type'] == 'too_low').sum()
            too_high = (df[f'{attr}_outlier_type'] == 'too_high').sum()
            print(f"    - Too low: {too_low:,} ({too_low/outliers*100:.1f}%)")
            print(f"    - Too high: {too_high:,} ({too_high/outliers*100:.1f}%)")
    
    # Overall summary
    total_listings = len(df)
    assigned_segments = df['segment_key'].notna().sum()
    
    print(f"\n{'='*60}")
    print(f"Overall Summary:")
    print(f"Total listings: {total_listings:,}")
    print(f"Assigned to segments: {assigned_segments:,} ({assigned_segments/total_listings*100:.1f}%)")
    print(f"{'='*60}")
    
    return df


def get_segment_summary(df: pd.DataFrame, attribute: str = 'price_to_sqft') -> pd.DataFrame:
    """
    Get summary statistics for each segment for a specific attribute.
    
    Args:
        df: DataFrame with segment assignments and outlier flags
        attribute: Attribute to summarize (default: 'price_to_sqft')
    
    Returns:
        DataFrame with segment-level statistics
    """
    outlier_col = f'is_{attribute}_outlier'
    
    # Check if outlier column exists, if not use default
    if outlier_col not in df.columns:
        outlier_col = 'is_price_to_sqft_outlier'
        attribute = 'price_to_sqft'
    
    agg_dict = {
        'segment_size': ('segment_key', 'size'),
        'outlier_count': (outlier_col, 'sum'),
        'outlier_pct': (outlier_col, 'mean'),
        'relaxation_level': ('relaxation_level', 'first')
    }
    
    # Add attribute-specific stats if column exists
    if attribute in df.columns:
        agg_dict.update({
            f'{attribute}_median': (attribute, 'median'),
            f'{attribute}_mean': (attribute, 'mean'),
            f'{attribute}_std': (attribute, 'std'),
            f'{attribute}_min': (attribute, 'min'),
            f'{attribute}_max': (attribute, 'max')
        })
    
    summary = df[df['segment_key'].notna()].groupby('segment_key').agg(**agg_dict).reset_index()
    
    summary['outlier_pct'] = summary['outlier_pct'] * 100
    
    return summary.sort_values('segment_size', ascending=False)
