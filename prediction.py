# Index(['no_of_dependents', 'education', 'self_employed', 'income_annum',
#        'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
#        'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'],
#       dtype='object')

import joblib
import pandas as pd

# Load the model and column order
saved_data = joblib.load('random_forest_model.joblib')
model = saved_data['model']
columns = saved_data['columns']
scaler = saved_data['scaler']

# Create a DataFrame for new data, ensuring the same column order
new_data = pd.DataFrame([
    {
        'no_of_dependents': 1,
        'education': 1,  # Graduate
        'self_employed': 0,  # No
        'income_annum': 50000,
        'loan_amount': 5000,
        'loan_term': 12,
        'cibil_score': 750,
        'residential_assets_value': 200000,
        'commercial_assets_value': 150000,
        'luxury_assets_value': 50000,
        'bank_asset_value': 100000
    }
])

# Ensure the new data is in the same order as the training data
new_data = new_data[columns]

# Scale the numerical features
numerical_features = ['income_annum', 'bank_asset_value', 'loan_amount', 'cibil_score', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value']
new_data[numerical_features] = scaler.transform(new_data[numerical_features])

# Predict using the loaded model
prediction = model.predict(new_data)
print('Prediction:', prediction)
