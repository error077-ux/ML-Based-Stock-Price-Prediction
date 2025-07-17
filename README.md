# ML-Based-Stock-Price-Prediction
This project implements a machine learning-based regression model to predict stock market trends using historical data. It features a Python backend (Flask) for model training and prediction, and a simple HTML/CSS/JavaScript frontend for user interaction and visualizing predictions.

# 🚀 Features
- Data Loading & Preprocessing: Handles historical stock data, including feature engineering (lag features, moving averages).
- Machine Learning Model: Utilizes a RandomForestRegressor from scikit-learn for robust prediction.
- Real-time Prediction API: A Flask backend provides an API endpoint (/predict) to receive input features and return a predicted stock price.
- Interactive Frontend: An HTML/CSS/JavaScript web interface allows users to input stock parameters and visualize the predicted price along with a historical chart.
- Plot Visualization: Generates a Matplotlib chart on the backend and sends it to the frontend for display, showing historical prices and the new prediction.

# 🛠️ Technologies Used
**Python:**
- pandas: For data manipulation and analysis.
- numpy: For numerical operations.
- scikit-learn: For machine learning model (RandomForestRegressor), data splitting, and scaling.
- matplotlib: For generating plots.
- Flask: Web framework for building the backend API.
- Flask-CORS: For handling Cross-Origin Resource Sharing, allowing frontend-backend communication.
- io, base64: For handling image data in memory.

**Frontend:**
- HTML5: Structure of the web page.
- CSS3 (Tailwind CSS): Styling and responsive design.
- JavaScript (ES6+): Frontend logic, interacting with the backend API.

# 📂 Project Structure
```
stock_predictor/
├── app.py                # Flask backend: ML model training, prediction API, plot generation
├── index.html            # Frontend: User interface, sends data to backend, displays results and chart
├── stock_data.csv        # (Example) Placeholder for your historical stock data (e.g., AAPL.csv)
└── stock_predictor.py    # (Optional) Original standalone script for local testing/plotting
```

# ⚙️ Setup Instructions

Follow these steps to get the project up and running on your local machine.

**Prerequisites**
```
Python 3.x: Download and install from python.org.
- Important: During Python installation, ensure you check "Add Python to PATH".
Visual Studio Code (VS Code): Recommended IDE. Download from code.visualstudio.com.
- Install the Python extension by Microsoft in VS Code.
- (Optional, but recommended for frontend) Install the Live Server extension by Ritwick Dey in VS Code.
```
**1. Clone the Repository**
- If you have this project as a Git repository, clone it:
```
git clone https://github.com/error077-ux/ML-Based-Stock-Price-Prediction.git
cd ML-Based-Stock-Price-Prediction
```
- If you're setting it up manually, simply create a folder named stock_predictor on your desktop (e.g., C:\Users\YourUser\Desktop\stock_predictor) and place the app.py and index.html files inside it.

**2. Install Python Dependencies**

- Open your Command Prompt (CMD) or VS Code Terminal and navigate to your project directory:
```
cd C:\Users\PC\OneDrive\Desktop\stock_predictor
```

- Then, install the required Python libraries:
```
python -m pip install pandas numpy scikit-learn matplotlib Flask Flask-Cors
```

**3. Obtain Historical Stock Data**

Your model needs real data!
- Go to a financial data source like Yahoo Finance.
- Search for a stock (e.g., AAPL, GOOG, RELIANCE.NS).
- Go to the "Historical Data" tab, select a time period, and click "Download".
- Save the downloaded CSV file (e.g., AAPL.csv) directly into your stock_predictor project folder.
```
**Note**
Important: Open app.py in VS Code. Locate the pd.read_csv('stock_data.csv', ...) line and ensure it matches the exact filename of your downloaded CSV file (e.g., if you downloaded AAPL.csv and renamed it to stock_data.csv, it's already correct. If you used sampleniftyfifty.csv, ensure that line is updated to sampleniftyfifty.csv).
```

# 🚀 Running the Application

You need to run the backend and frontend separately.
**1. Start the Backend (Flask API)**
- Open a new Command Prompt (CMD) or VS Code Terminal and navigate to your project directory:
```
cd C:\Users\PC\OneDrive\Desktop\stock_predictor
```
- Run the Flask application:
```
python app.py
```
- You should see output indicating that the Flask server is running on http://127.0.0.1:5000. Keep this terminal window open; the server must be running for the frontend to work.

**2. Open the Frontend (HTML)**
- Now, open your index.html file in a web browser:

```
- Using VS Code Live Server
- In VS Code, open the Explorer sidebar.
- Right-click on index.html.
- Select "Open with Live Server".
```
This will open the page in your default browser, usually at http://127.0.0.1:5500/index.html.

**3. Interact with the Application**
- Once the index.html page is open in your browser, you will see input fields for various stock parameters.
- Enter numerical values into all the fields.
- Click the "Predict Next Day Close Price" button.
- The predicted price will appear.
- A new "Show Chart" button will appear.
- Click the "Show Chart" button to display the Matplotlib plot, showing the historical data and your new prediction.
---
