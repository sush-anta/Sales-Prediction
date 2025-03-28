import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset, fixing encoding issues
df = pd.read_csv("car_purchasing.csv", encoding='latin-1')

# Ensure column names are clean
df.columns = df.columns.str.strip()

# Drop unnecessary columns
df = df.drop(columns=["customer name", "customer e-mail", "country"])

# Standardize numerical features (except target variable)
scaler = StandardScaler()
numeric_cols = ["age", "annual Salary", "credit card debt", "net worth"]
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])  # Scale only features

# Define features (X) and target variable (y)
X = df.drop(columns=["car purchase amount"])
y = df["car purchase amount"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate error metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print evaluation results
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2 Score): {r2:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.xlabel("Actual Car Purchase Amount")
plt.ylabel("Predicted Car Purchase Amount")
plt.title("Actual vs Predicted Sales")
plt.show()

# Feature Importance
feature_importance = model.feature_importances_
plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance, y=X.columns)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Sales Prediction")
plt.show()

# Save Model & Scaler
joblib.dump(model, "sales_prediction_model.pkl")
joblib.dump(scaler, "scaler.pkl")
