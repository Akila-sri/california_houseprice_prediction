import streamlit as st
import pickle
import numpy as np

# Load trained model & scaler
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Streamlit App
st.title("🏡 House Price Prediction App")
st.write("Enter house details to predict the price.")

# Input fields
med_inc = st.number_input("Median Income (10,000s)", min_value=0.0, max_value=15.0, value=5.0)
house_age = st.number_input("House Age", min_value=0, max_value=100, value=20)
rooms = st.number_input("Avg Rooms per Household", min_value=1.0, max_value=10.0, value=5.0)
bedrooms = st.number_input("Avg Bedrooms per Household", min_value=1.0, max_value=5.0, value=1.0)
population = st.number_input("Population in Area", min_value=100.0, max_value=50000.0, value=1000.0)
occupants = st.number_input("Avg Occupants per Household", min_value=1.0, max_value=10.0, value=3.0)
latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=35.0)
longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-120.0)

# Predict button
if st.button("Predict Price"):
    # Prepare input data
    input_data = np.array([[med_inc, house_age, rooms, bedrooms, population, occupants, latitude, longitude]])
    
    # Transform input using the correct scaler
    input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.success(f"Predicted House Price: ${prediction:,.2f}")
