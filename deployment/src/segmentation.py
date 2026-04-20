from typing import Dict, Any
import pandas as pd
from deployment.configs.constants import (
    TOP_RESIDENTIAL_TYPES,
    REST_OF_HOUSING_TYPES,
    APPLY_LOCATION_ROLLUP
)


def create_segment_key(row: pd.Series, include_bedrooms: bool = True,
                       location_level: str = 'location_id') -> str:
    parts = [
        str(row.get(location_level, '')),
        str(row.get('housing_type_name', '')),
        str(row.get('offering_type_name', '')),
        str(row.get('property_price_type_name', ''))
    ]

    if include_bedrooms:
        parts.append(str(row.get('bedrooms', '')))

    return '|'.join(parts)


def assign_segment_from_lookup(row: pd.Series, lookup_dict: Dict,
                               location_tree_lookup: pd.DataFrame = None) -> Dict[str, Any]:
    result = {
        'segment_key': None,
        'unseen': True,
        'relaxation_level': None,
        'segment_count': None,
        'price_median': None,
        'property_sqft_median': None,
        'price_to_sqft_median': None
    }

    housing_type = row.get('housing_type_name', '')
    is_top_residential = housing_type in TOP_RESIDENTIAL_TYPES
    is_rest_housing = housing_type in REST_OF_HOUSING_TYPES

    location_levels_dict = {}
    if is_rest_housing and location_tree_lookup is not None and not location_tree_lookup.empty:
        location_id = row.get('location_id')
        if location_id is not None:
            location_row = location_tree_lookup[location_tree_lookup['location_id'] == location_id]
            if not location_row.empty:
                location_levels_dict = location_row.iloc[0].to_dict()

    if is_top_residential:
        for include_bedrooms in [True, False]:
            if not result['unseen']:
                break

            segment_key = create_segment_key(row, include_bedrooms, 'location_id')

            if segment_key in lookup_dict:
                result['segment_key'] = segment_key
                result['unseen'] = False

                segment_data = lookup_dict[segment_key]
                result['relaxation_level'] = segment_data.get('relaxation_level')
                result['segment_count'] = segment_data.get('segment_count')
                result['price_median'] = segment_data.get('price_median')
                result['property_sqft_median'] = segment_data.get('property_sqft_median')
                result['price_to_sqft_median'] = segment_data.get('price_to_sqft_median')
                break

    elif is_rest_housing:
        if APPLY_LOCATION_ROLLUP:
            location_levels = ['location_id', 'location_lvl_3_id', 'location_lvl_2_id',
                              'location_lvl_1_id', 'location_lvl_0_id']
        else:
            location_levels = ['location_id']

        for location_level in location_levels:
            if not result['unseen']:
                break

            location_value = row.get(location_level) or location_levels_dict.get(location_level)

            if location_value is None or pd.isna(location_value):
                continue

            temp_row = row.copy()
            temp_row[location_level] = location_value

            segment_key = create_segment_key(temp_row, include_bedrooms=False, location_level=location_level)

            if segment_key in lookup_dict:
                result['segment_key'] = segment_key
                result['unseen'] = False

                segment_data = lookup_dict[segment_key]
                result['relaxation_level'] = segment_data.get('relaxation_level')
                result['segment_count'] = segment_data.get('segment_count')
                result['price_median'] = segment_data.get('price_median')
                result['property_sqft_median'] = segment_data.get('property_sqft_median')
                result['price_to_sqft_median'] = segment_data.get('price_to_sqft_median')
                break

    if result['unseen']:
        result['relaxation_level'] = 'unseen'

    return result
