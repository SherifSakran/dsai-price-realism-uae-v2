import sys
import threading
from datetime import datetime
from io import BytesIO
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import ClientError
from deployment.configs.feedback_config import (
    CX_FEEDBACK_S3_BUCKET,
    CX_FEEDBACK_S3_KEY,
    FEEDBACK_REFRESH_SECONDS,
    FEEDBACK_SAFETY_NET_FACTOR
)
from deployment.src.utils import normalize_property_type, format_bedrooms, get_offering_type_name
from deployment.src.segmentation import create_segment_key
from deployment.configs.constants import PRICE_TYPE_TO_NAME


feedback_lookup = {
    'by_listing_id': {},
    'by_segment_key': {},
}
feedback_last_update_ts: Optional[datetime] = None
_feedback_lock = threading.Lock()
_feedback_updating = False


def build_feedback_lookup_table(disputed_df: pd.DataFrame) -> Dict[str, Any]:
    by_listing_id = {}
    by_segment_key = {}

    for _, listing in disputed_df.iterrows():
        listing_id = str(listing.get('property_listing_id', ''))
        price = pd.to_numeric(listing.get('price', None), errors='coerce')
        sqft = pd.to_numeric(listing.get('property_sqft', None), errors='coerce')

        if pd.isna(price) or pd.isna(sqft) or sqft == 0:
            continue

        price_to_sqft = price / sqft

        property_type = normalize_property_type(str(listing.get('property_type', listing.get('housing_type_name', ''))))
        category = str(listing.get('category', listing.get('offering_type_name', ''))).lower()
        price_type = str(listing.get('price_type', listing.get('property_price_type_name', ''))).lower()

        if category in ['residential sale', 'residential rent', 'commercial sale', 'commercial rent']:
            offering_type_name = category.title()
        else:
            offering_type_name = get_offering_type_name(category, price_type)

        price_type_name = PRICE_TYPE_TO_NAME.get(price_type, str(listing.get('property_price_type_name', '')))

        bedrooms = format_bedrooms(listing.get('bedrooms'))

        row = pd.Series({
            'location_id': str(listing.get('location_id', '')),
            'housing_type_name': property_type,
            'offering_type_name': offering_type_name,
            'property_price_type_name': price_type_name,
            'bedrooms': bedrooms,
        })

        segment_key = create_segment_key(row, include_bedrooms=True, location_level='location_id')
        segment_key_no_bed = create_segment_key(row, include_bedrooms=False, location_level='location_id')

        entry = {
            'segment_key': segment_key,
            'segment_key_no_bed': segment_key_no_bed,
            'price_to_sqft_median': price_to_sqft,
        }

        by_listing_id[listing_id] = entry

        for sk in [segment_key, segment_key_no_bed]:
            if sk not in by_segment_key:
                by_segment_key[sk] = []
            by_segment_key[sk].append(price_to_sqft)

    by_segment_key_median = {}
    for sk, values in by_segment_key.items():
        by_segment_key_median[sk] = float(np.median(values))

    return {
        'by_listing_id': by_listing_id,
        'by_segment_key': by_segment_key_median,
    }


def _load_feedback_from_s3():
    global feedback_lookup, feedback_last_update_ts
    try:
        s3 = boto3.client('s3')
        print(f"[FEEDBACK] Fetching s3://{CX_FEEDBACK_S3_BUCKET}/{CX_FEEDBACK_S3_KEY}", file=sys.stderr)
        resp = s3.get_object(Bucket=CX_FEEDBACK_S3_BUCKET, Key=CX_FEEDBACK_S3_KEY)
        body = resp['Body'].read()
        disputed_df = pd.read_parquet(BytesIO(body))
        print(f"[FEEDBACK] Loaded {len(disputed_df):,} disputed listings from S3", file=sys.stderr)

        new_lookup = build_feedback_lookup_table(disputed_df)

        with _feedback_lock:
            feedback_lookup = new_lookup
            feedback_last_update_ts = datetime.now()

        print(f"[FEEDBACK] Built feedback lookup: {len(new_lookup['by_listing_id'])} listings, "
              f"{len(new_lookup['by_segment_key'])} segments", file=sys.stderr)

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code in ('NoSuchKey', 'NoSuchBucket', '404'):
            print("[FEEDBACK] cx_feedback_loop.parquet not found in S3, starting with empty feedback",
                  file=sys.stderr)
            with _feedback_lock:
                feedback_lookup = {'by_listing_id': {}, 'by_segment_key': {}}
                feedback_last_update_ts = datetime.now()
        else:
            print(f"[FEEDBACK] S3 ClientError ({error_code}): {e}, keeping current feedback",
                  file=sys.stderr)
            if feedback_last_update_ts is None:
                with _feedback_lock:
                    feedback_last_update_ts = datetime.now()
    except Exception as e:
        print(f"[FEEDBACK] Failed to load feedback from S3: {e}, keeping current feedback", file=sys.stderr)
        if feedback_last_update_ts is None:
            with _feedback_lock:
                feedback_last_update_ts = datetime.now()


def _background_refresh_feedback():
    global _feedback_updating
    try:
        _load_feedback_from_s3()
    finally:
        _feedback_updating = False


def maybe_refresh_feedback():
    global _feedback_updating
    if feedback_last_update_ts is None:
        return
    elapsed = (datetime.now() - feedback_last_update_ts).total_seconds()
    if elapsed >= FEEDBACK_REFRESH_SECONDS and not _feedback_updating:
        _feedback_updating = True
        print("[FEEDBACK] Feedback stale, triggering background refresh", file=sys.stderr)
        thread = threading.Thread(target=_background_refresh_feedback, daemon=True)
        thread.start()


def build_feedback_response(
    price_to_sqft_median: float,
    property_sqft: float,
) -> Dict[str, Any]:
    pts_lower = (1.0 - FEEDBACK_SAFETY_NET_FACTOR) * price_to_sqft_median
    pts_upper = (1.0 + FEEDBACK_SAFETY_NET_FACTOR) * price_to_sqft_median
    lower_bound = pts_lower * property_sqft
    upper_bound = pts_upper * property_sqft

    return {
        "valid_size": True,
        "unseen": False,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "segment_abs_pct_error": None,
    }


def submit_feedback(feedback_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Submit feedback by appending to S3 parquet file and updating in-memory lookup.
    
    Follows the same structure as export_disputed_listings.py:
    All columns are object (str) dtype except price and property_sqft which are float64.
    """
    global feedback_lookup, feedback_last_update_ts
    
    try:
        s3 = boto3.client('s3')
        
        # Load existing feedback data from S3
        try:
            print(f"[FEEDBACK] Loading existing feedback from s3://{CX_FEEDBACK_S3_BUCKET}/{CX_FEEDBACK_S3_KEY}", file=sys.stderr)
            resp = s3.get_object(Bucket=CX_FEEDBACK_S3_BUCKET, Key=CX_FEEDBACK_S3_KEY)
            existing_df = pd.read_parquet(BytesIO(resp['Body'].read()))
            print(f"[FEEDBACK] Loaded {len(existing_df):,} existing disputed listings", file=sys.stderr)
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ('NoSuchKey', 'NoSuchBucket', '404'):
                print("[FEEDBACK] No existing feedback file found, creating new one", file=sys.stderr)
                existing_df = pd.DataFrame()
            else:
                raise
        
        # Map input fields to match export_disputed_listings.py output format
        property_type = normalize_property_type(str(feedback_data['property_type']))
        category = str(feedback_data['category']).lower()
        price_type = str(feedback_data['price_type']).lower()
        offering_type_name = get_offering_type_name(category, price_type)
        property_price_type_name = PRICE_TYPE_TO_NAME.get(price_type, price_type.title())
        
        # Build row matching the exact schema from export_disputed_listings.py
        new_row = pd.DataFrame([{
            'property_listing_id': str(feedback_data.get('property_listing_id', '')),
            'location_id': str(feedback_data['location_id']),
            'housing_type_name': str(property_type),
            'offering_type_name': str(offering_type_name),
            'property_price_type_name': str(property_price_type_name),
            'bedrooms': str(feedback_data.get('bedrooms', '')),
            'price': float(feedback_data['valid_price']),
            'property_sqft': float(feedback_data['valid_property_sqft']),
        }])
        
        # Ensure existing df columns match the same types before concat
        if not existing_df.empty:
            for col in existing_df.columns:
                if col not in ('price', 'property_sqft'):
                    existing_df[col] = existing_df[col].astype(str)
        
        updated_df = pd.concat([existing_df, new_row], ignore_index=True)
        
        # Remove duplicates based on property_listing_id
        listing_id = str(feedback_data.get('property_listing_id', ''))
        if listing_id:
            updated_df = updated_df.drop_duplicates(subset=['property_listing_id'], keep='last')
        
        print(f"[FEEDBACK] Updated dataframe: {len(updated_df):,} listings", file=sys.stderr)
        
        # Write back to S3
        buffer = BytesIO()
        updated_df.to_parquet(buffer, index=False)
        buffer.seek(0)
        
        s3.put_object(
            Bucket=CX_FEEDBACK_S3_BUCKET,
            Key=CX_FEEDBACK_S3_KEY,
            Body=buffer.getvalue()
        )
        print(f"[FEEDBACK] Successfully wrote updated feedback to S3", file=sys.stderr)
        
        # Rebuild in-memory lookup
        new_lookup = build_feedback_lookup_table(updated_df)
        
        with _feedback_lock:
            feedback_lookup = new_lookup
            feedback_last_update_ts = datetime.now()
        
        print(f"[FEEDBACK] Updated in-memory lookup: {len(new_lookup['by_listing_id'])} listings, "
              f"{len(new_lookup['by_segment_key'])} segments", file=sys.stderr)
        
        return {
            "action": "submit_feedback",
            "status": "success",
            "message": "Feedback submitted successfully",
            "property_listing_id": listing_id,
            "feedback_listings": len(feedback_lookup['by_listing_id']),
            "feedback_segments": len(feedback_lookup['by_segment_key']),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"[FEEDBACK] Failed to submit feedback: {e}", file=sys.stderr)
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        raise Exception(f"Failed to submit feedback: {str(e)}")
