from typing import Dict, Any
import pandas as pd
from deployment.configs.constants import (
    REST_OF_HOUSING_TYPES,
    SEGMENT_SIZE_THRESHOLD,
    PTS_MULTIPLIER,
    PTS_LARGER_MULTIPLIER,
    LOCATION_MULTIPLIERS
)
from deployment.configs.feedback_config import FEEDBACK_MULTIPLIER_BOOST


def determine_multiplier(segment_count: int, relaxation_level: str, housing_type: str) -> float:
    if relaxation_level and relaxation_level.endswith('_cx'):
        base_level = relaxation_level[:-3]
        base_multiplier = determine_multiplier(segment_count, base_level, housing_type)
        return base_multiplier * FEEDBACK_MULTIPLIER_BOOST

    if housing_type in REST_OF_HOUSING_TYPES:
        return LOCATION_MULTIPLIERS.get(relaxation_level, 6)

    if segment_count >= SEGMENT_SIZE_THRESHOLD or relaxation_level == 'without_bedrooms':
        return PTS_LARGER_MULTIPLIER
    else:
        return PTS_MULTIPLIER


def check_outlier(value: float, median: float, multiplier: float) -> Dict[str, Any]:
    if pd.isna(value) or pd.isna(median) or median == 0:
        return {
            'is_outlier': False,
            'outlier_type': 'not_outlier',
            'deviation_ratio': None
        }

    high_threshold = multiplier * median
    low_threshold = (1.0 / multiplier) * median

    is_high = value > high_threshold
    is_low = value < low_threshold

    if is_high:
        deviation_ratio = value / median
        return {
            'is_outlier': True,
            'outlier_type': 'high',
            'deviation_ratio': deviation_ratio
        }
    elif is_low:
        deviation_ratio = median / value
        return {
            'is_outlier': True,
            'outlier_type': 'low',
            'deviation_ratio': deviation_ratio
        }
    else:
        deviation_ratio = value / median
        return {
            'is_outlier': False,
            'outlier_type': 'not_outlier',
            'deviation_ratio': deviation_ratio
        }
