import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# ADD THESE IMPORTS
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
# Load cleaned dataset
df = pd.read_csv("data/cleaned_car_data.csv")

# Create car age feature
current_year = 2026
df["car_age"] = current_year - df["year"]

# Remove original year column
df.drop(columns=["year"], inplace=True)

# Display updated dataset
print("UPDATED DATASET")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN INFORMATION")
print(df.info())
# Separate features and target

X = df.drop(columns=["selling_price"])
y = df["selling_price"]

# Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAIN-TEST SPLIT")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)
# Identify categorical and numerical columns

categorical_columns = [
    "name",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

numerical_columns = [
    "km_driven",
    "car_age"
]

# Create preprocessing transformer

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("numerical", "passthrough", numerical_columns)
    ]
)

# Fit on training data and transform both datasets

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

print("\nAFTER CATEGORICAL ENCODING")
print("Encoded training shape:", X_train_encoded.shape)
print("Encoded testing shape:", X_test_encoded.shape)
# Train Linear Regression model

linear_model = LinearRegression()

linear_model.fit(X_train_encoded, y_train)

# Predict on test data

y_pred = linear_model.predict(X_test_encoded)

# Evaluate the model

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nLINEAR REGRESSION RESULTS")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)
# Train Random Forest Regressor

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_encoded, y_train)

# Predict on test data

rf_pred = rf_model.predict(X_test_encoded)

# Evaluate Random Forest

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\nRANDOM FOREST RESULTS")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)
# Model Comparison

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [mae, rf_mae],
    "RMSE": [rmse, rf_rmse],
    "R2 Score": [r2, rf_r2]
})

print("\nMODEL COMPARISON")
print(results)

# Select best model based on R2 Score
best_model_name = results.loc[results["R2 Score"].idxmax(), "Model"]

print("\nBEST MODEL:", best_model_name)
# Select the best model based on R2 Score

if r2 >= rf_r2:
    best_model = linear_model
    best_model_name = "Linear Regression"
else:
    best_model = rf_model
    best_model_name = "Random Forest"

print("\nSELECTED MODEL:", best_model_name)

# Save model and preprocessing transformer together

joblib.dump(
    {
        "model": best_model,
        "preprocessor": preprocessor
    },
    "used_car_price_model.pkl"
)

print("\nMODEL SAVED SUCCESSFULLY!")