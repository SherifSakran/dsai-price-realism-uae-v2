import os

CX_FEEDBACK_S3_BUCKET = os.getenv(
    'CX_FEEDBACK_S3_BUCKET', 'dsai-price-realism-staging')
CX_FEEDBACK_S3_KEY = os.getenv(
    'CX_FEEDBACK_S3_KEY', 'uae/feedback/disputed_listings_details.parquet')
FEEDBACK_MULTIPLIER_BOOST = float(os.getenv('FEEDBACK_MULTIPLIER_BOOST', '1.5'))
FEEDBACK_REFRESH_SECONDS = int(os.getenv('FEEDBACK_REFRESH_SECONDS', '3600'))
FEEDBACK_SAFETY_NET_FACTOR = float(os.getenv('FEEDBACK_SAFETY_NET_FACTOR', '0.2'))
