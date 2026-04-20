import os

MODEL_PATH = os.getenv('MODEL_PATH', '/opt/ml/model')

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

SEGMENT_SIZE_THRESHOLD = 20
PTS_MULTIPLIER = 3
PTS_LARGER_MULTIPLIER = 4

APPLY_LOCATION_ROLLUP = False

LOCATION_MULTIPLIERS = {
    'location_id': 6,
    'location_lvl_3': 7,
    'location_lvl_2': 8,
    'location_lvl_1': 9,
    'location_lvl_0': 10
}

PRICE_TYPE_TO_NAME = {
    "sale": "Sale",
    "monthly": "Monthly",
    "yearly": "Yearly",
    "daily": "Daily",
    "weekly": "Weekly"
}

OFFERING_TYPE_ID_TO_NAME = {
    '1': 'Residential Sale',
    '2': 'Residential Rent',
    '3': 'Commercial Sale',
    '4': 'Commercial Rent'
}

HOUSING_TYPE_ID_MAP = {
    '1': 'Apartment', '35': 'Villa', '22': 'Townhouse', '4': 'Office Space',
    '44': 'Factory', '5': 'Land', '24': 'Duplex', '20': 'Penthouse', '46': 'Restaurant',
    '21': 'Shop', '27': 'Retail', '13': 'Warehouse', '45': 'Hotel & Hotel Apartment',
    '10': 'Whole Building', '50': 'Farm', '42': 'Compound', '47': 'Storage',
    '11': 'Labor Camp', '18': 'Full Floor', '48': 'Business Centre', '12': 'Show Room',
    '30': 'Bulk Sale Unit', '34': 'Bulk Rent Unit', '43': 'Staff Accommodation',
    '29': 'Half Floor', '31': 'Bungalow', '49': 'Co-working space', '19': 'Plot'
}

PRICE_TYPE_MAP = {
    "1": "Sale", 
    "3": "Monthly", 
    "4": "Yearly", 
    "6": "Daily", 
    "7": "Weekly"
}
