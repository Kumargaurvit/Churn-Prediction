import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Loading the Model from pickle file
model = load_model("Models/model.h5") 

# Loading the Scaler and Encoder pickle files
with open("Models/scaler.pkl","rb") as file:
    scaler = pickle.load(file)

with open("Models/label_encoder.pkl","rb") as file:
    label_encoder = pickle.load(file)

with open("Models/one_hot_encoder.pkl","rb") as file:
    one_hot_encoder = pickle.load(file)

st.title("Customer Churn Prediction")

# Taking required information as input
geography = st.selectbox("Geography", one_hot_encoder.categories_[0])
gender = st.selectbox("Gender", label_encoder.classes_)
age = st.slider("Age", 18, 95)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Preparing input data
input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [label_encoder.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

# Encoding Geography column and Combining with the Input Data
geography_encoded = one_hot_encoder.transform([[geography]]).toarray()

geography_encoded = pd.DataFrame(geography_encoded,columns=one_hot_encoder.get_feature_names_out())

input_data = pd.concat([input_data,geography_encoded],axis=1)

# Scaling the Data
input_data_scaled = scaler.transform(input_data)

# Churn Prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f"Churn Probability : {prediction_proba}")

if prediction_proba > 0.5:
    st.write("Customer is likely to Churn!")
else:
    st.write("Customer is not likely to Churn!")