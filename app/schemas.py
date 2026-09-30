from pydantic import BaseModel, Field
from typing import Literal


class HouseFeatures(BaseModel):
    State: Literal[
        "Maharashtra", "Karnataka", "Delhi", "Tamil Nadu", "Gujarat",
        "Telangana", "Uttar Pradesh", "Rajasthan", "West Bengal", "Punjab"
    ] = Field(..., example="Karnataka")
    City: str = Field(..., example="Bengaluru")
    Property_Type: Literal[
        "Apartment", "Villa", "Independent House", "Plot", "Studio"
    ] = Field(..., example="Apartment")
    BHK: int = Field(..., ge=1, le=10, example=3)
    Size_in_SqFt: int = Field(..., ge=100, le=20000, example=1200)
    Year_Built: int = Field(..., ge=1950, le=2024, example=2015)
    Furnished_Status: Literal["Furnished", "Semi-Furnished", "Unfurnished"] = Field(..., example="Semi-Furnished")
    Floor_No: int = Field(..., ge=0, le=60, example=5)
    Total_Floors: int = Field(..., ge=1, le=80, example=12)
    Age_of_Property: int = Field(..., ge=0, le=74, example=9)
    Nearby_Schools: int = Field(..., ge=0, le=20, example=3)
    Nearby_Hospitals: int = Field(..., ge=0, le=20, example=2)
    Public_Transport_Accessibility: int = Field(..., ge=1, le=10, example=7)
    Parking_Space: int = Field(..., ge=0, le=1, example=1)
    Security: int = Field(..., ge=0, le=1, example=1)
    Amenities: int = Field(..., ge=0, le=10, example=5)
    Facing: Literal["North", "South", "East", "West", "North-East", "South-West"] = Field(..., example="North-East")
    Owner_Type: Literal["Owner", "Builder", "Dealer"] = Field(..., example="Builder")
    Availability_Status: Literal["Ready to Move", "Under Construction", "Pre-Launch"] = Field(..., example="Ready to Move")

    class Config:
        json_schema_extra = {
            "example": {
                "State": "Karnataka",
                "City": "Bengaluru",
                "Property_Type": "Apartment",
                "BHK": 3,
                "Size_in_SqFt": 1200,
                "Year_Built": 2015,
                "Furnished_Status": "Semi-Furnished",
                "Floor_No": 5,
                "Total_Floors": 12,
                "Age_of_Property": 9,
                "Nearby_Schools": 3,
                "Nearby_Hospitals": 2,
                "Public_Transport_Accessibility": 7,
                "Parking_Space": 1,
                "Security": 1,
                "Amenities": 5,
                "Facing": "North-East",
                "Owner_Type": "Builder",
                "Availability_Status": "Ready to Move"
            }
        }


class PredictionResponse(BaseModel):
    predicted_price_lakhs: float
    model_version: str = "1.0.0"
