import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Set the backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, request, jsonify
from flask_cors import CORS
import io # For handling in-memory bytes
import base64 # For encoding image to base64

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Global Variables for Model and Scalers ---
model = None
scaler_X = None
scaler_y = None
features = ['Close_Lag1', 'Volume_Lag1', 'MA5', 'MA10', 'Open', 'High', 'Low']
target = 'Close'
df_global = None # To store the full dataframe for plotting historical data

# --- Function to Load Data and Train Model ---
def train_model():
    global model, scaler_X, scaler_y, df_global # Declare globals to modify them

    print("--- Starting Model Training for Flask App ---")

    # Load your actual stock data from a CSV file
    # Make sure 'samplecsv.csv' is in the same directory as app.py
    try:
        # CHANGED: Loading 'samplecsv.csv'
        df = pd.read_csv('sampleniftyfifty.csv', index_col='Date', parse_dates=True)
        # Sort by date to ensure correct time-series order
        df.sort_index(inplace=True)
        # Optional: Use 'Adj Close' if available and preferred
        # If your CSV has 'Adj Close' and you want to use it, uncomment the next line:
        # df['Close'] = df['Adj Close']
        print(f"Successfully loaded data from samplecsv.csv. Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: 'samplecsv.csv' not found. Please ensure the CSV file is in the same directory as app.py.")
        print("Using dummy data for now. Please download your stock data and place it in the project folder.")
        # Fallback to dummy data if file not found (useful for testing)
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
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

    # Store the full dataframe globally for plotting
    df_global = df.copy()

    # Feature Engineering
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Volume_Lag1'] = df['Volume'].shift(1)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df.dropna(inplace=True)

    X = df[features]
    y = df[target]

    # Initialize and fit scalers
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    X_scaled = pd.DataFrame(X_scaled, columns=features, index=X.index)
    y_scaled = pd.Series(y_scaled.flatten(), name=target, index=y.index)

    # Splitting Data (using 80/20 time-series split)
    train_size = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[0:train_size], X_scaled[train_size:len(X_scaled)]
    y_train, y_test = y_scaled[0:train_size], y_scaled[train_size:len(y_scaled)]

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("--- Model Training Complete for Flask App ---")

# --- API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    # Validate incoming data has all required features
    if not all(feature in data for feature in features):
        return jsonify({"error": "Missing one or more required features"}), 400

    # Create a DataFrame from the incoming JSON data
    input_data = pd.DataFrame([data], columns=features)

    try:
        # Scale the input data using the trained scaler
        input_scaled = scaler_X.transform(input_data)

        # Make prediction using the trained model
        prediction_scaled = model.predict(input_scaled)

        # Inverse transform the prediction to original price scale
        predicted_price = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]

        # --- Generate Plot ---
        plt.figure(figsize=(12, 6))
        # Plot historical data
        plt.plot(df_global.index, df_global['Close'], label='Historical Close Price', color='blue', alpha=0.7)

        # Get the last date from the historical data for plotting the prediction
        last_historical_date = df_global.index[-1]
        # Create a date for the predicted point (e.g., next day)
        # Note: This assumes daily data. Adjust freq if your data is different.
        predicted_date = last_historical_date + pd.Timedelta(days=1)

        # Plot the predicted point
        plt.plot(predicted_date, predicted_price, marker='o', markersize=8, color='red', label='Predicted Next Day Close')

        plt.title('Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout() # Adjust layout to prevent labels from overlapping

        # Save plot to a BytesIO object
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0) # Rewind to the beginning of the buffer
        plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close() # Close the plot to free up memory

        return jsonify({'prediction': predicted_price, 'plot_image': plot_base64})
    except Exception as e:
        # Log the error for debugging
        print(f"Error during prediction or plotting: {e}")
        return jsonify({"error": f"Prediction or plotting failed: {str(e)}"}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    train_model() # Train the model once when the app starts
    app.run(debug=True, port=5000)
