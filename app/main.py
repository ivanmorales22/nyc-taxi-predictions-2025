import streamlit as st
import requests
import json

st.write("""
# Application to predict the time for the NYC taxi trips
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    PU = st.sidebar.text_input("PU Location ID", "80")
    DO = st.sidebar.text_input("DO Location ID", "60")
    trip_distance = st.sidebar.number_input("Trip Distance", value=10, min_value=1, max_value=100)

    input_dict = {
        'PULocationID': PU,
        'DOLocationID': DO,
        'trip_distance': trip_distance,
    }

    return input_dict


input_dict = user_input_features()

if st.button('Predict'):
    response = requests.post(
        url="http://127.0.0.1:8000/api/v1/predict",
        data=json.dumps(input_dict)
    )

    st.write(f"El tiempo estimado del viaje es: {response.json()['prediction']} minutos")
