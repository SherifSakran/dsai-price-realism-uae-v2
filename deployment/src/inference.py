import sys
from typing import Dict, Any
import pandas as pd
from deployment.src.utils import normalize_property_type, format_bedrooms, get_offering_type_name
from deployment.src.segmentation import create_segment_key, assign_segment_from_lookup
from deployment.src.outlier_detection import determine_multiplier, check_outlier
from deployment.src.feedback_loop import build_feedback_response
from deployment.configs.constants import PRICE_TYPE_TO_NAME
from deployment.configs.feedback_config import FEEDBACK_MULTIPLIER_BOOST


def process_single_request(data: Dict[str, Any], lookup_dict: Dict, artifacts: Dict, feedback_lookup: Dict = None) -> Dict[str, Any]:
    if feedback_lookup is None:
        feedback_lookup = {'by_listing_id': {}, 'by_segment_key': {}}
    property_sqft = data.get('property_sqft', 0)

    property_type = normalize_property_type(data.get('property_type', ''))
    category = data.get('category', '').lower()
    price_type = data.get('price_type', '').lower()

    row = pd.Series({
        'location_id': data.get('location_id'),
        'housing_type_name': property_type,
        'offering_type_name': get_offering_type_name(category, price_type),
        'property_price_type_name': PRICE_TYPE_TO_NAME.get(price_type),
        'bedrooms': format_bedrooms(data.get('bedrooms')),
        'property_sqft': property_sqft,
        'price': data.get('price')
    })

    row['price_to_sqft'] = row['price'] / row['property_sqft']

    property_listing_id = str(data.get('property_listing_id', ''))
    feedback_hit_type = None

    if property_listing_id and property_listing_id in feedback_lookup['by_listing_id']:
        entry = feedback_lookup['by_listing_id'][property_listing_id]
        feedback_hit_type = 'listing_id'
        print(f"[FEEDBACK] Hit by listing_id={property_listing_id}, "
              f"pts_median={entry['price_to_sqft_median']:.4f}", file=sys.stderr)
        return build_feedback_response(
            price_to_sqft_median=entry['price_to_sqft_median'],
            property_sqft=row['property_sqft'],
        )

    segment_key_with_bed = create_segment_key(row, include_bedrooms=True, location_level='location_id')
    segment_key_no_bed = create_segment_key(row, include_bedrooms=False, location_level='location_id')
    feedback_segment_key = None
    feedback_pts_median = None
    for sk in [segment_key_with_bed, segment_key_no_bed]:
        if sk in feedback_lookup['by_segment_key']:
            feedback_hit_type = 'segment_key'
            feedback_segment_key = sk
            feedback_pts_median = feedback_lookup['by_segment_key'][sk]
            print(f"[FEEDBACK] Hit by segment_key={sk}, will boost multiplier by "
                  f"{FEEDBACK_MULTIPLIER_BOOST}x and use feedback pts_median={feedback_pts_median:.4f}", file=sys.stderr)
            break

    location_tree_lookup = artifacts.get('location_tree_lookup', pd.DataFrame())
    segment_info = assign_segment_from_lookup(row, lookup_dict, location_tree_lookup)
    
    if feedback_hit_type == 'segment_key' and feedback_pts_median is not None:
        segment_info['price_to_sqft_median'] = feedback_pts_median
        print(f"[FEEDBACK] Overriding price_to_sqft_median with feedback value: {feedback_pts_median:.4f}", file=sys.stderr)

    if segment_info['unseen']:
        print(f"[INFO] Unseen segment for listing", file=sys.stderr)
        return {
            "valid_size": True,
            "unseen": True,
            "lower_bound": None,
            "upper_bound": None,
            "segment_abs_pct_error": None
        }

    multiplier = determine_multiplier(
        segment_info['segment_count'],
        segment_info['relaxation_level'],
        row['housing_type_name']
    )

    if feedback_hit_type == 'segment_key':
        multiplier = multiplier * FEEDBACK_MULTIPLIER_BOOST

    sqft_outlier_result = {'is_outlier': False, 'outlier_type': 'not_outlier', 'deviation_ratio': None}
    if segment_info['property_sqft_median'] is not None:
        sqft_outlier_result = check_outlier(
            row['property_sqft'],
            segment_info['property_sqft_median'],
            multiplier
        )
        print(f"[DEBUG] SQFT check: value={row['property_sqft']}, median={segment_info['property_sqft_median']}, "
              f"multiplier={multiplier}, is_outlier={sqft_outlier_result['is_outlier']}, "
              f"type={sqft_outlier_result['outlier_type']}", file=sys.stderr)
        print(f"[INFO] SQFT bounds: lb={1.0/multiplier*segment_info['property_sqft_median']}, "
              f"ub={multiplier*segment_info['property_sqft_median']}")

    price_to_sqft_outlier_result = {'is_outlier': False, 'outlier_type': 'not_outlier', 'deviation_ratio': None}
    if row['price_to_sqft'] is not None and segment_info['price_to_sqft_median'] is not None:
        price_to_sqft_outlier_result = check_outlier(
            row['price_to_sqft'],
            segment_info['price_to_sqft_median'],
            multiplier
        )
        print(f"[DEBUG] PTS check: value={row['price_to_sqft']}, median={segment_info['price_to_sqft_median']}, "
              f"multiplier={multiplier}, is_outlier={price_to_sqft_outlier_result['is_outlier']}, "
              f"type={price_to_sqft_outlier_result['outlier_type']}", file=sys.stderr)

    price_outlier_result = {'is_outlier': False, 'outlier_type': 'not_outlier', 'deviation_ratio': None}
    if row['price'] is not None and segment_info['price_median'] is not None:
        price_outlier_result = check_outlier(
            row['price'],
            segment_info['price_median'],
            multiplier
        )
        print(f"[DEBUG] PRICE check: value={row['price']}, median={segment_info['price_median']}, "
              f"multiplier={multiplier}, is_outlier={price_outlier_result['is_outlier']}, "
              f"type={price_outlier_result['outlier_type']}", file=sys.stderr)

    valid_size = not (sqft_outlier_result['is_outlier'] and price_to_sqft_outlier_result['is_outlier'])
    print(f"[DEBUG] valid_size={valid_size} (sqft_outlier={sqft_outlier_result['is_outlier']}, "
          f"pts_outlier={price_to_sqft_outlier_result['is_outlier']})", file=sys.stderr)

    if not valid_size:
        return {
            "valid_size": valid_size,
            "unseen": False,
            "lower_bound": None,
            "upper_bound": None,
            "segment_abs_pct_error": None
        }

    is_outlier = False
    outlier_type = 'not_outlier'
    deviation_ratio = None

    if price_to_sqft_outlier_result['is_outlier']:
        is_outlier = True
        outlier_type = 'price_to_sqft_outlier'
        deviation_ratio = price_to_sqft_outlier_result['deviation_ratio']
    elif price_outlier_result['is_outlier']:
        is_outlier = True
        outlier_type = 'price_outlier'
        deviation_ratio = price_outlier_result['deviation_ratio']
    elif sqft_outlier_result['is_outlier']:
        is_outlier = True
        outlier_type = 'sqft_outlier'
        deviation_ratio = sqft_outlier_result['deviation_ratio']

    print(f"[INFO] Internal fields - is_outlier: {is_outlier}, outlier_type: {outlier_type}, "
          f"deviation_ratio: {deviation_ratio}, segment_key: {segment_info['segment_key']}, "
          f"segment_count: {segment_info['segment_count']}, relaxation_level: {segment_info['relaxation_level']}, "
          f"price_median: {segment_info['price_median']}, property_sqft_median: {segment_info['property_sqft_median']}, "
          f"price_to_sqft_median: {segment_info['price_to_sqft_median']}, multiplier_used: {multiplier}",
          file=sys.stderr)

    lower_bound = None
    upper_bound = None

    if segment_info['price_to_sqft_median'] is not None:
        pts_lower = (1.0 / multiplier) * segment_info['price_to_sqft_median']
        pts_upper = multiplier * segment_info['price_to_sqft_median']
        lower_bound = pts_lower * row['property_sqft']
        upper_bound = pts_upper * row['property_sqft']


    return {
        "valid_size": valid_size,
        "unseen": False,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "segment_abs_pct_error": None
    }
