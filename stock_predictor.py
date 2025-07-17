import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor # A robust regression model
from sklearn.metrics import mean_squared_error

# --- 1. Data Loading ---
# For demonstration, we'll create a dummy dataset.
# In a real project, you would load your data from a CSV or other source:
# df = pd.read_csv('your_stock_data.csv', index_col='Date', parse_dates=True)

print("--- 1. Data Loading ---")
# Create a dummy dataset for demonstration purposes
# Simulate stock prices over 100 days
np.random.seed(42) # for reproducibility
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
# Simulate a general upward trend with some noise and seasonality
base_price = 100
prices = base_price + np.cumsum(np.random.randn(100) * 0.5) + np.sin(np.arange(100) / 10) * 5
volumes = np.random.randint(100000, 500000, 100)

df = pd.DataFrame({
    'Close': prices,
    'Volume': volumes,
    'Open': prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
    'High': prices * (1 + np.random.uniform(0.005, 0.015, 100)),
    'Low': prices * (1 - np.random.uniform(0.005, 0.015, 100))
}, index=dates)

print("Original Data Head:")
print(df.head())
print("\nData Info:")
df.info()

# --- 2. Data Preprocessing & Feature Engineering ---
print("\n--- 2. Data Preprocessing & Feature Engineering ---")

# Handle missing values (if any) - for this dummy data, there are none,
# but in real data, you might use df.fillna(method='ffill') or df.dropna()
print("Checking for missing values:")
print(df.isnull().sum())

# Create lag features (previous day's close price)
# This is a common feature for time series prediction
df['Close_Lag1'] = df['Close'].shift(1)
df['Volume_Lag1'] = df['Volume'].shift(1)

# Create moving averages as features
df['MA5'] = df['Close'].rolling(window=5).mean()
df['MA10'] = df['Close'].rolling(window=10).mean()

# Drop rows with NaN values created by shifting and rolling operations
df.dropna(inplace=True)
print("\nData Head after Feature Engineering and NaN removal:")
print(df.head())

# Define features (X) and target (y)
features = ['Close_Lag1', 'Volume_Lag1', 'MA5', 'MA10', 'Open', 'High', 'Low']
target = 'Close'

X = df[features]
y = df[target]

# Scale features using MinMaxScaler
# Scaling is important for many ML algorithms
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler() # Scale target if using algorithms sensitive to target scale

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)) # Reshape for scaler

# Convert back to DataFrame for easier handling, preserving column names
X_scaled = pd.DataFrame(X_scaled, columns=features, index=X.index)
y_scaled = pd.Series(y_scaled.flatten(), name=target, index=y.index)

print("\nScaled Features Head:")
print(X_scaled.head())

# --- 3. Splitting Data into Training and Testing Sets ---
print("\n--- 3. Splitting Data ---")
# Use a time-series split (important for stock data)
# We'll use the first 80% for training and the last 20% for testing
train_size = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[0:train_size], X_scaled[train_size:len(X_scaled)]
y_train, y_test = y_scaled[0:train_size], y_scaled[train_size:len(y_scaled)]

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

# --- 4. Model Training ---
print("\n--- 4. Model Training ---")
# Initialize a Random Forest Regressor model
# Random Forest is good for its robustness and ability to handle non-linear relationships
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Train the model
print("Training the Random Forest Regressor...")
model.fit(X_train, y_train)
print("Model training complete.")

# --- 5. Prediction ---
print("\n--- 5. Prediction ---")
# Make predictions on the scaled test data
y_pred_scaled = model.predict(X_test)

# Inverse transform the predictions and actual values to their original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_original = scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).flatten()

# Create a DataFrame for easy comparison
predictions_df = pd.DataFrame({
    'Actual': y_test_original,
    'Predicted': y_pred
}, index=y_test.index)

print("\nSample Predictions vs. Actuals:")
print(predictions_df.head())

# --- 6. Evaluation ---
print("\n--- 6. Evaluation ---")
# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# --- 7. Visualization ---
print("\n--- 7. Visualization ---")
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], label='Historical Close Price', color='blue', alpha=0.7)
plt.plot(predictions_df.index, predictions_df['Actual'], label='Actual Test Price', color='green', linewidth=2)
plt.plot(predictions_df.index, predictions_df['Predicted'], label='Predicted Test Price', color='red', linestyle='--', linewidth=2)
plt.title('Stock Price Prediction using Random Forest Regressor')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()

print("\nProject initialization complete. You can now replace the dummy data with your actual stock data.")
print("Consider trying different models, features, and hyperparameter tuning for better accuracy.")
