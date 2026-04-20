from typing import Union
from deployment.configs.constants import PRICE_TYPE_TO_NAME


def normalize_property_type(property_type: str) -> str:
    if not property_type:
        return property_type

    property_type_mapping = {
        'apartment': 'Apartment',
        'villa': 'Villa',
        'townhouse': 'Townhouse',
        'office space': 'Office Space',
        'factory': 'Factory',
        'land': 'Land',
        'duplex': 'Duplex',
        'penthouse': 'Penthouse',
        'restaurant': 'Restaurant',
        'shop': 'Shop',
        'retail': 'Retail',
        'warehouse': 'Warehouse',
        'hotel & hotel apartment': 'Hotel & Hotel Apartment',
        'whole building': 'Whole Building',
        'farm': 'Farm',
        'compound': 'Compound',
        'storage': 'Storage',
        'labor camp': 'Labor Camp',
        'full floor': 'Full Floor',
        'business centre': 'Business Centre',
        'show room': 'Show Room',
        'bulk sale unit': 'Bulk Sale Unit',
        'bulk rent unit': 'Bulk Rent Unit',
        'staff accommodation': 'Staff Accommodation',
        'half floor': 'Half Floor',
        'bungalow': 'Bungalow',
        'co-working space': 'Co-working space',
        'plot': 'Plot'
    }

    return property_type_mapping.get(property_type.lower(), property_type)


def format_bedrooms(bedrooms: Union[str, None]) -> str:
    if bedrooms is None or bedrooms == "":
        return "N/A"

    bedrooms_str = str(bedrooms).strip()

    if bedrooms_str.lower() in ["n/a", "na", "none"]:
        return "N/A"

    if bedrooms_str.endswith(" Bed") or bedrooms_str == "7+ Beds" or bedrooms_str in ["Studio", "Unknown"]:
        return bedrooms_str

    if bedrooms_str.lower() in ["studio", "0"]:
        return "Studio"

    if "7+" in bedrooms_str or bedrooms_str.lower().startswith("7+"):
        return "7+ Beds"

    try:
        bed_num = int(float(bedrooms_str))
        if bed_num >= 8:
            return "7+ Beds"
        elif bed_num == 7:
            return "7 Bed"
        elif bed_num == 0:
            return "Studio"
        else:
            return f"{bed_num} Bed"
    except (ValueError, TypeError):
        return bedrooms_str


def get_offering_type_name(category: str, price_type: str) -> str:
    if category == "residential":
        if price_type == "sale":
            return "Residential Sale"
        else:
            return "Residential Rent"
    else:
        if price_type == "sale":
            return "Commercial Sale"
        else:
            return "Commercial Rent"
