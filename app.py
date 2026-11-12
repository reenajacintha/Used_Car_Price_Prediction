import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessor
saved_objects = joblib.load("used_car_price_model.pkl")

model = saved_objects["model"]
preprocessor = saved_objects["preprocessor"]

# Page configuration
st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Used Car Price Prediction")
st.write("Enter your car details to estimate its resale price.")

# Load dataset for dropdown options
df = pd.read_csv("data/cleaned_car_data.csv")

# User inputs
st.subheader("Car Details")

car_name = st.selectbox(
    "Car Name",
    sorted(df["name"].unique())
)

km_driven = st.number_input(
    "Kilometres Driven",
    min_value=1,
    max_value=1000000,
    value=30000
)

fuel = st.selectbox(
    "Fuel Type",
    sorted(df["fuel"].unique())
)

seller_type = st.selectbox(
    "Seller Type",
    sorted(df["seller_type"].unique())
)

transmission = st.selectbox(
    "Transmission",
    sorted(df["transmission"].unique())
)

owner = st.selectbox(
    "Owner Type",
    sorted(df["owner"].unique())
)

car_age = st.number_input(
    "Car Age (Years)",
    min_value=0,
    max_value=50,
    value=5
)

# Prediction
if st.button("Predict Car Price"):
    input_data = pd.DataFrame([{
        "name": car_name,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "car_age": car_age
    }])

    transformed_data = preprocessor.transform(input_data)

    predicted_price = model.predict(transformed_data)[0]

    st.success(
        f"Estimated Resale Price: ₹{predicted_price:,.0f}"
    )