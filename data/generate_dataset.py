"""
Generate synthetic India House Price dataset with ~250,000 rows and 22 columns.
This mimics realistic distributions for an Indian real estate dataset.
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 250000

STATES = ["Maharashtra", "Karnataka", "Delhi", "Tamil Nadu", "Gujarat",
          "Telangana", "Uttar Pradesh", "Rajasthan", "West Bengal", "Punjab"]

CITIES_BY_STATE = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik"],
    "Karnataka": ["Bengaluru", "Mysuru", "Hubli", "Mangaluru"],
    "Delhi": ["New Delhi", "Dwarka", "Rohini", "Noida"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Khammam"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Agra", "Varanasi"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota"],
    "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Siliguri"],
    "Punjab": ["Chandigarh", "Ludhiana", "Amritsar", "Jalandhar"],
}

CITY_MULTIPLIER = {
    "Mumbai": 3.5, "New Delhi": 3.0, "Bengaluru": 2.8, "Hyderabad": 2.2,
    "Chennai": 2.0, "Pune": 2.0, "Ahmedabad": 1.6, "Kolkata": 1.5,
    "Noida": 1.7, "Chandigarh": 1.6,
}

PROPERTY_TYPES = ["Apartment", "Villa", "Independent House", "Plot", "Studio"]
FURNISHED_STATUS = ["Furnished", "Semi-Furnished", "Unfurnished"]
OWNER_TYPES = ["Owner", "Builder", "Dealer"]
AVAILABILITY = ["Ready to Move", "Under Construction", "Pre-Launch"]
FACING_OPTS = ["North", "South", "East", "West", "North-East", "South-West"]

def generate():
    states = np.random.choice(STATES, N, p=[0.18,0.15,0.12,0.10,0.09,0.09,0.08,0.07,0.07,0.05])
    cities, localities = [], []
    city_mults = []

    for s in states:
        city = np.random.choice(CITIES_BY_STATE[s])
        cities.append(city)
        localities.append(f"Locality_{np.random.randint(1, 50)}")
        city_mults.append(CITY_MULTIPLIER.get(city, 1.4))

    city_mults = np.array(city_mults)

    bhk = np.random.choice([1, 2, 3, 4, 5], N, p=[0.10, 0.35, 0.35, 0.15, 0.05])
    size_base = bhk * 450 + np.random.normal(0, 150, N)
    size = np.clip(size_base, 300, 8000).astype(int)

    year_built = np.random.randint(1980, 2024, N)
    age = 2024 - year_built
    floor_no = np.random.randint(0, 30, N)
    total_floors = floor_no + np.random.randint(1, 15, N)
    nearby_schools = np.random.randint(0, 10, N)
    nearby_hospitals = np.random.randint(0, 8, N)
    public_transport = np.random.randint(1, 10, N)
    parking = np.random.choice([0, 1], N, p=[0.3, 0.7])
    security = np.random.choice([0, 1], N, p=[0.2, 0.8])
    amenities = np.random.randint(0, 10, N)

    prop_types = np.random.choice(PROPERTY_TYPES, N, p=[0.55, 0.15, 0.15, 0.10, 0.05])
    prop_mult = np.where(prop_types == "Villa", 1.5,
                np.where(prop_types == "Plot", 0.7,
                np.where(prop_types == "Studio", 0.8, 1.0)))

    furnished = np.random.choice(FURNISHED_STATUS, N, p=[0.30, 0.45, 0.25])
    furn_mult = np.where(furnished == "Furnished", 1.15,
                np.where(furnished == "Semi-Furnished", 1.05, 1.0))

    # Price formula: base price driven by size, city, BHK, age
    base_price = (size * 4.5 * city_mults * prop_mult * furn_mult
                  + bhk * 3
                  + amenities * 2
                  + security * 1.5
                  + parking * 2
                  - age * 0.3
                  + nearby_schools * 0.5
                  + public_transport * 0.3)
    base_price = base_price / 100  # scale to lakhs
    noise = np.random.normal(0, base_price * 0.08)
    price = np.clip(base_price + noise, 5, 50000).round(2)

    df = pd.DataFrame({
        "ID": range(1, N + 1),
        "State": states,
        "City": cities,
        "Locality": localities,
        "Property_Type": prop_types,
        "BHK": bhk,
        "Size_in_SqFt": size,
        "Price_in_Lakhs": price,
        "Year_Built": year_built,
        "Furnished_Status": furnished,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Age_of_Property": age,
        "Nearby_Schools": nearby_schools,
        "Nearby_Hospitals": nearby_hospitals,
        "Public_Transport_Accessibility": public_transport,
        "Parking_Space": parking,
        "Security": security,
        "Amenities": amenities,
        "Facing": np.random.choice(FACING_OPTS, N),
        "Owner_Type": np.random.choice(OWNER_TYPES, N, p=[0.45, 0.35, 0.20]),
        "Availability_Status": np.random.choice(AVAILABILITY, N, p=[0.60, 0.30, 0.10]),
    })
    return df

if __name__ == "__main__":
    print("Generating dataset...")
    df = generate()
    out = os.path.join(os.path.dirname(__file__), "house_prices.csv")
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows to {out}")
    print(df.head(3))
    print(df.describe())
