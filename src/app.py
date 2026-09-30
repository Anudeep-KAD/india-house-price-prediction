import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="India House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


# ============================================================
# HEADER
# ============================================================

st.title("🏠 India House Price Prediction")
st.markdown(
    "### Predict the estimated price of a residential property in India"
)

st.divider()


# ============================================================
# PROPERTY LOCATION
# ============================================================

st.header("📍 Property Location")

col1, col2, col3 = st.columns(3)

with col1:
    state = st.selectbox(
        "State",
        [
            "Odisha",
            "Tamil Nadu",
            "West Bengal",
            "Gujarat",
            "Delhi",
            "Telangana",
            "Maharashtra",
            "Punjab",
            "Uttar Pradesh",
            "Uttarakhand",
            "Assam",
            "Kerala",
            "Jharkhand",
            "Andhra Pradesh",
            "Chhattisgarh",
            "Madhya Pradesh",
            "Karnataka",
            "Rajasthan",
            "Bihar",
            "Haryana"
        ]
    )

with col2:
    city = st.text_input(
        "City",
        placeholder="Enter city"
    )

with col3:
    locality = st.text_input(
        "Locality",
        placeholder="Enter locality"
    )


# ============================================================
# PROPERTY DETAILS
# ============================================================

st.header("🏡 Property Details")

col1, col2, col3, col4 = st.columns(4)

with col1:
    property_type = st.selectbox(
        "Property Type",
        [
            "Villa",
            "Independent House",
            "Apartment"
        ]
    )

with col2:
    bhk = st.number_input(
        "BHK",
        min_value=1,
        max_value=5,
        value=2,
        step=1
    )

with col3:
    size = st.number_input(
        "Size (Sq Ft)",
        min_value=500,
        max_value=5000,
        value=1500,
        step=50
    )

with col4:
    year_built = st.number_input(
        "Year Built",
        min_value=1990,
        max_value=2023,
        value=2015,
        step=1
    )


# Dataset used 2025 as the reference year for Age_of_Property.
age_of_property = 2025 - year_built

st.info(f"Calculated Property Age: {age_of_property} years")


# ============================================================
# FLOOR DETAILS
# ============================================================

st.header("🏢 Floor Details")

col1, col2, col3 = st.columns(3)

with col1:
    floor_no = st.number_input(
        "Floor Number",
        min_value=0,
        max_value=30,
        value=1,
        step=1
    )

with col2:
    total_floors = st.number_input(
        "Total Floors",
        min_value=1,
        max_value=30,
        value=5,
        step=1
    )

with col3:
    nearby_schools = st.number_input(
        "Nearby Schools",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )

nearby_hospitals = st.number_input(
    "Nearby Hospitals",
    min_value=1,
    max_value=10,
    value=5,
    step=1
)


# ============================================================
# PROPERTY FEATURES
# ============================================================

st.header("✨ Property Features")

col1, col2, col3 = st.columns(3)

with col1:
    furnished_status = st.selectbox(
        "Furnished Status",
        [
            "Unfurnished",
            "Semi-furnished",
            "Furnished"
        ]
    )

with col2:
    transport = st.selectbox(
        "Public Transport Accessibility",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

with col3:
    parking = st.selectbox(
        "Parking Space",
        [
            "No",
            "Yes"
        ]
    )


col1, col2, col3 = st.columns(3)

with col1:
    security = st.selectbox(
        "Security",
        [
            "No",
            "Yes"
        ]
    )

with col2:
    facing = st.selectbox(
        "Facing",
        [
            "North",
            "South",
            "East",
            "West"
        ]
    )

with col3:
    owner_type = st.selectbox(
        "Owner Type",
        [
            "Owner",
            "Broker",
            "Builder"
        ]
    )


availability_status = st.selectbox(
    "Availability Status",
    [
        "Ready_to_Move",
        "Under_Construction"
    ]
)


# ============================================================
# AMENITIES
# ============================================================

st.header("🏊 Amenities")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    has_pool = st.checkbox("🏊 Pool")

with col2:
    has_gym = st.checkbox("🏋️ Gym")

with col3:
    has_garden = st.checkbox("🌳 Garden")

with col4:
    has_clubhouse = st.checkbox("🏢 Clubhouse")

with col5:
    has_playground = st.checkbox("⚽ Playground")


# ============================================================
# PREDICTION
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict House Price",
    type="primary",
    use_container_width=True
)


if predict_button:

    # Validate floor information
    if floor_no > total_floors:
        st.error(
            "Floor Number cannot be greater than Total Floors."
        )
        st.stop()

    # Validate city
    if not city.strip():
        st.error("Please enter the city.")
        st.stop()

    # Validate locality
    if not locality.strip():
        st.error("Please enter the locality.")
        st.stop()


    # ========================================================
    # CREATE INPUT DATAFRAME
    # ========================================================

    input_data = pd.DataFrame({
        "State": [state],
        "City": [city.strip()],
        "Locality": [locality.strip()],
        "Property_Type": [property_type],
        "BHK": [bhk],
        "Size_in_SqFt": [size],
        "Year_Built": [year_built],
        "Furnished_Status": [furnished_status],
        "Floor_No": [floor_no],
        "Total_Floors": [total_floors],
        "Age_of_Property": [age_of_property],
        "Nearby_Schools": [nearby_schools],
        "Nearby_Hospitals": [nearby_hospitals],
        "Public_Transport_Accessibility": [transport],
        "Parking_Space": [parking],
        "Security": [security],
        "Facing": [facing],
        "Owner_Type": [owner_type],
        "Availability_Status": [availability_status],
        "Has_Pool": [int(has_pool)],
        "Has_Gym": [int(has_gym)],
        "Has_Garden": [int(has_garden)],
        "Has_Clubhouse": [int(has_clubhouse)],
        "Has_Playground": [int(has_playground)]
    })


    # ========================================================
    # PREPROCESS INPUT
    # ========================================================

    input_encoded = preprocessor.transform(input_data)


    # ========================================================
    # MAKE PREDICTION
    # ========================================================

    prediction = model.predict(input_encoded)[0]


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.success("Prediction completed successfully! 🎉")

    st.markdown("## 🏠 Estimated House Price")

    st.metric(
        label="Predicted Price",
        value=f"₹ {prediction:.2f} Lakhs"
    )

    st.caption(
        "The prediction is generated using the trained Random Forest model."
    )