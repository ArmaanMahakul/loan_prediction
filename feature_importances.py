import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the saved model
saved_data = joblib.load('random_forest_model.joblib')
model = saved_data['model']
columns = saved_data['columns']

# Extract feature importances
importances = model.feature_importances_

# Create a DataFrame for better visualization
feature_importances = pd.DataFrame({
    'Feature': columns,
    'Importance': importances
})

# Sort the features by importance
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Print feature importances
print(feature_importances)

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances in Random Forest Model')
plt.gca().invert_yaxis()  # To display the most important feature at the top
plt.show()
