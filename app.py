import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the trained model and the scaler
saved_data = joblib.load('random_forest_model.joblib')
model = saved_data['model']

st.title("Loan Prediction App")

# Input fields
income_annum = st.number_input('Annual Income', min_value=0)
bank_asset_value = st.number_input('Bank Asset Value', min_value=0)
loan_amount = st.number_input('Loan Amount', min_value=0)
cibil_score = st.slider('CIBIL Score', min_value=300, max_value=900, value=300)
residential_assets_value = st.number_input('Residential Assets Value', min_value=0)
commercial_assets_value = st.number_input('Commercial Assets Value', min_value=0)
luxury_assets_value = st.number_input('Luxury Assets Value', min_value=0)
no_of_dependents = st.slider('No of Dependents', min_value=0, max_value=5, value=1, step=1)

# Radio buttons for education
education = st.radio(
    'Education',
    options=['Graduate', 'Not Graduate'],
    index=0
)

if education == 'Graduate':
    education = 1
else:
    education = 0

# Radio buttons for self-employed status
self_employed = st.radio(
    'Self Employed',
    options=['Yes', 'No'],
    index=1
)

if self_employed == 'Yes':
    self_employed = 1
else:
    self_employed = 0

# Radio buttons for loan term
loan_term = st.slider('Loan Term', min_value=2, max_value=20, step=2)

# Prepare the input data for prediction
input_data = np.array([[no_of_dependents, education, self_employed, income_annum,
                        loan_amount, loan_term, cibil_score, residential_assets_value,
                        commercial_assets_value, luxury_assets_value, bank_asset_value]])

scaler = MinMaxScaler()
input_data = scaler.fit_transform(input_data)

# Placeholder for prediction result
st.write("### Prediction Result")

# Perform the prediction when the button is clicked
if st.button('Predict'):
    prediction = model.predict(input_data)
    if prediction == 1:
        st.write("Loan Approved")
    else:
        st.write("Loan Rejected")


