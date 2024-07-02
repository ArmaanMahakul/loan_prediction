import joblib
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler

# Data Loading
df = pd.read_csv('dataset.csv')
df = df.drop(['loan_id'], axis=1)
df.columns = df.columns.str.strip()
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

numerical_features = ['income_annum', 'bank_asset_value', 'loan_amount', 'cibil_score', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value']
categorical_features = ['no_of_dependents', 'loan_term', 'education', 'self_employed']

# Define mapping dictionary
mapping1 = {'Graduate': 1, 'Not Graduate': 0}
mapping2 = {'Yes': 1, 'No': 0}
mapping3 = {'Approved': 1, 'Rejected': 0}

# Apply mapping using map function
df['education'] = df['education'].map(mapping1)
df['self_employed'] = df['self_employed'].map(mapping2)
df['loan_status'] = df['loan_status'].map(mapping3)

# Splitting the data
target = 'loan_status'
x = df.drop(columns=[target])
y = df[target]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=27)

for train_index, test_index in sss.split(x, y):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Scaling the data
scaler = MinMaxScaler()
x_train.loc[:, numerical_features] = scaler.fit_transform(x_train[numerical_features])
x_test.loc[:, numerical_features] = scaler.transform(x_test[numerical_features])

model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=27)  # Limit depth and number of trees
model.fit(x_train, y_train)
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
report = classification_report(y_test, y_pred_test)
confusion = confusion_matrix(y_test, y_pred_test)

print(f'Training Accuracy: {accuracy_train}')
print(f'Test Accuracy: {accuracy_test}')
print('Confusion Matrix:')
print(confusion)
print('Classification Report:')
print(report)
print(x.columns)

# Save the model and the column order
joblib.dump({'model': model, 'columns': x.columns.tolist(), 'scaler': scaler}, 'random_forest_model.joblib')
