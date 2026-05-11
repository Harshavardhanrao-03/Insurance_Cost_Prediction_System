import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
df = pd.read_csv('insurance.csv')

# Create cleaned dataframe
df_cleaned = df.copy()

# Encode categorical columns
le = LabelEncoder()

df_cleaned['is_female'] = le.fit_transform(
    df_cleaned['sex']
)

df_cleaned['is_smoker'] = le.fit_transform(
    df_cleaned['smoker']
)

# BMI category feature engineering
df_cleaned['bmi_category'] = pd.cut(
    df_cleaned['bmi'],
    bins=[0, 18.5, 25, 30, 100],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese']
)

# One-hot encoding
df_cleaned = pd.get_dummies(
    df_cleaned,
    columns=['region', 'bmi_category'],
    drop_first=False
)

# Final selected features
final_df = df_cleaned[[
    'age',
    'is_female',
    'bmi',
    'children',
    'is_smoker',
    'charges',
    'region_southeast',
    'bmi_category_Obese'
]]

# Features and target
X = final_df.drop('charges', axis=1)

y = final_df['charges']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("R2 Score:", r2)

print("RMSE:", rmse)

# Save model
with open('insurance_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully!")