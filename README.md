# 🏠 India House Price Prediction

A machine learning based web application that predicts residential property prices in India using property characteristics such as location, size, BHK, property type, furnishing status, floor details, nearby facilities, amenities, and availability status.

## 🚀 Live Application

The application is deployed using Streamlit Community Cloud.

> 🔗 **Live App:** https://india-house-price-prediction-ahxcffhr28bfvnly84qjd7.streamlit.app/

## 📌 Project Overview

The goal of this project is to build a complete machine learning pipeline for predicting house prices.

The project includes:

- Data cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Categorical encoding
- Train-test splitting
- Regression model training
- Model evaluation
- Model serialization
- Streamlit web application
- Cloud deployment

## 📊 Dataset

The dataset contains **250,000 property records** and **22 original columns**.

Important attributes include:

- State
- City
- Locality
- Property Type
- BHK
- Size in SqFt
- Price in Lakhs
- Year Built
- Furnished Status
- Floor Number
- Total Floors
- Age of Property
- Nearby Schools
- Nearby Hospitals
- Public Transport Accessibility
- Parking Space
- Security
- Amenities
- Facing
- Owner Type
- Availability Status

## 🧹 Data Cleaning

The following preprocessing steps were performed:

- Missing-value checking
- Duplicate-row checking
- Duplicate-ID checking
- Numerical-value validation
- Invalid floor relationship correction
- Age consistency verification
- Target-variable validation
- Price-per-square-foot consistency analysis
- Outlier analysis

The final cleaned dataset contains:

**250,000 rows × 26 columns**

## ⚙️ Feature Engineering

The `Amenities` column was converted into five binary features:

- `Has_Pool`
- `Has_Gym`
- `Has_Garden`
- `Has_Clubhouse`
- `Has_Playground`

The original `Amenities` column was then removed from the modeling dataset.

## 🤖 Machine Learning Models

The following regression models were evaluated:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor
4. XGBoost Regressor

### Model Evaluation

Models were evaluated using:

- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- R² Score

### Best Baseline Model

The Random Forest model achieved the best baseline RMSE among the tested models.

However, the dataset showed very weak relationships between the available features and `Price_in_Lakhs`, resulting in R² values close to zero.

This result is reported honestly rather than using target leakage to artificially increase model performance.

## 🔐 Target Leakage Prevention

`Price_per_SqFt` was not used as a model input because it is directly derived from the target price and property size.

Using it as an input would introduce target leakage and produce misleading model performance.

## 🌐 Web Application

The application was developed using **Streamlit**.

Users can enter:

- State
- City
- Locality
- Property Type
- BHK
- Size
- Year Built
- Floor Number
- Total Floors
- Nearby Schools
- Nearby Hospitals
- Furnishing Status
- Public Transport Accessibility
- Parking
- Security
- Facing
- Owner Type
- Availability
- Amenities

The application then predicts the estimated property price in Indian Lakhs.

## 📁 Project Structure

```text
India-House-Price-Prediction/
│
├── data/
│   ├── raw/
│   │   └── india_housing_prices.csv
│   └── processed/
│       └── india_housing_prices_cleaned.csv
│
├── models/
│   ├── feature_info.json
│   ├── preprocessor.pkl
│   └── random_forest_model.pkl
│
├── notebooks/
│   └── 01_data_analysis.ipynb
│
├── src/
│   └── app.py
│
├── .gitignore
├── requirements.txt
└── README.md