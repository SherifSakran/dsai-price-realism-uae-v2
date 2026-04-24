from typing import Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class PropertyTypeEnum(str, Enum):
    APARTMENT = "Apartment"
    VILLA = "Villa"
    TOWNHOUSE = "Townhouse"
    OFFICE_SPACE = "Office Space"
    FACTORY = "Factory"
    LAND = "Land"
    DUPLEX = "Duplex"
    PENTHOUSE = "Penthouse"
    RESTAURANT = "Restaurant"
    SHOP = "Shop"
    RETAIL = "Retail"
    WAREHOUSE = "Warehouse"
    HOTEL = "Hotel & Hotel Apartment"
    WHOLE_BUILDING = "Whole Building"
    FARM = "Farm"
    COMPOUND = "Compound"
    STORAGE = "Storage"
    LABOR_CAMP = "Labor Camp"
    FULL_FLOOR = "Full Floor"
    BUSINESS_CENTRE = "Business Centre"
    SHOW_ROOM = "Show Room"
    BULK_SALE_UNIT = "Bulk Sale Unit"
    BULK_RENT_UNIT = "Bulk Rent Unit"
    STAFF_ACCOMMODATION = "Staff Accommodation"
    HALF_FLOOR = "Half Floor"
    BUNGALOW = "Bungalow"
    COWORKING_SPACE = "Co-working space"
    PLOT = "Plot"


class PriceTypeEnum(str, Enum):
    SALE = "sale"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    WEEKLY = "weekly"
    DAILY = "daily"


class CategoryEnum(str, Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"


class CompletionStatusEnum(str, Enum):
    COMPLETED = "completed"
    OFF_PLAN = "off_plan"
    COMPLETED_PRIMARY = "completed_primary"
    OFF_PLAN_PRIMARY = "off_plan_primary"


class FurnishingTypeEnum(str, Enum):
    FURNISHED = "furnished"
    SEMI_FURNISHED = "semi-furnished"
    UNFURNISHED = "unfurnished"


class PredictionRequest(BaseModel):
    property_type: PropertyTypeEnum
    location_id: str
    price_type: PriceTypeEnum
    category: CategoryEnum
    property_sqft: float
    bedrooms: Union[str, None] = None
    bathrooms: Union[str, None] = None
    floor_number: Union[str, None] = None
    completion_status: Union[CompletionStatusEnum, None] = None
    furnishing_type: Union[FurnishingTypeEnum, None] = None
    property_parking: str = ""
    price: float
    property_listing_id: Union[str, None] = None

    @validator('property_type', pre=True)
    def normalize_property_type_input(cls, v):
        if isinstance(v, str):
            for enum_val in PropertyTypeEnum:
                if v.lower() == enum_val.value.lower():
                    return enum_val.value
        return v

    @validator('price_type', pre=True)
    def normalize_price_type_input(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v

    @validator('category', pre=True)
    def normalize_category_input(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v

    @validator('completion_status', pre=True)
    def normalize_completion_status_input(cls, v):
        if v is not None and isinstance(v, str):
            return v.lower()
        return v

    @validator('furnishing_type', pre=True)
    def normalize_furnishing_type_input(cls, v):
        if v is not None and isinstance(v, str):
            return v.lower()
        return v

    @validator('property_sqft')
    def validate_sqft(cls, v):
        if v <= 0:
            raise ValueError('property_sqft must be greater than 0')
        return v

    @validator('price')
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError('price must be greater than 0')
        return v


class PredictionResponse(BaseModel):
    valid_size: bool
    unseen: Union[bool, None]
    lower_bound: Union[float, None]
    upper_bound: Union[float, None]
    segment_abs_pct_error: Union[float, None]


class FeedbackSubmissionRequest(BaseModel):
    action: str = Field(default="submit_feedback")
    property_type: PropertyTypeEnum
    location_id: str
    price_type: PriceTypeEnum
    category: CategoryEnum
    valid_property_sqft: float
    bedrooms: Union[str, None] = None
    bathrooms: Union[str, None] = None
    floor_number: Union[str, None] = None
    completion_status: Union[CompletionStatusEnum, None] = None
    furnishing_type: Union[FurnishingTypeEnum, None] = None
    property_parking: str = ""
    valid_price: float
    property_listing_id: Union[str, None] = None

    @validator('property_type', pre=True)
    def normalize_property_type_input(cls, v):
        if isinstance(v, str):
            for enum_val in PropertyTypeEnum:
                if v.lower() == enum_val.value.lower():
                    return enum_val.value
        return v

    @validator('price_type', pre=True)
    def normalize_price_type_input(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v

    @validator('category', pre=True)
    def normalize_category_input(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v

    @validator('completion_status', pre=True)
    def normalize_completion_status_input(cls, v):
        if v is not None and isinstance(v, str):
            return v.lower()
        return v

    @validator('furnishing_type', pre=True)
    def normalize_furnishing_type_input(cls, v):
        if v is not None and isinstance(v, str):
            return v.lower()
        return v

    @validator('valid_property_sqft')
    def validate_sqft(cls, v):
        if v <= 0:
            raise ValueError('valid_property_sqft must be greater than 0')
        return v

    @validator('valid_price')
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError('valid_price must be greater than 0')
        return v
