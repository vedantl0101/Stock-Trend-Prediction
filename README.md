# Stock Trend Prediction


This project uses a Long Short-Term Memory (LSTM) neural network to predict stock prices based on historical data. The project is implemented using Python, leveraging libraries such as TensorFlow/Keras for building the model, Streamlit for the web interface, and yfinance for fetching stock data.

# Project Structure
app.py: Main application file that sets up the Streamlit interface, fetches data, preprocesses it, and displays the prediction results.
my_model.keras: Pre-trained LSTM model used for making predictions.
Stock Trend Prediction.ipynb: Jupyter Notebook containing the code for data processing, model training, and evaluation.
Features
Data Fetching: Fetches historical stock data using the yfinance library.
Data Visualization: Visualizes the stock's closing prices along with 100-day and 200-day moving averages.
Model Training: Trains an LSTM model to predict stock prices based on past data.
Prediction: Uses the trained model to predict future stock prices and compares them with actual prices.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/vedantl0101/stock-trend-prediction.git
cd stock-trend-prediction

Install the required packages:

bash
Copy code
pip install -r requirements.txt


Requirements - keras ,matplotlib ,numpy ,pandas ,scikit-learn ,streamlit ,yfinance


Ensure you have my_model.keras in the root directory of the project.

Usage
To run the application, use the following command:

bash
Copy code
streamlit run app.py
This will start the Streamlit server, and you can interact with the application via your web browser.

app.py
The app.py file contains the main code for the Streamlit application. It includes:

User Input: Accepts a stock ticker symbol from the user.
Data Fetching: Fetches historical stock data for the specified ticker.
Data Visualization: Plots the closing prices, and 100-day and 200-day moving averages.
Data Preparation: Splits the data into training and testing sets and scales the data.
Model Loading: Loads the pre-trained LSTM model.
Prediction: Makes predictions on the test data and visualizes the results.
Stock Trend Prediction.ipynb
The Jupyter Notebook includes:

Data Processing: Loads and preprocesses the data for model training.
Model Building: Defines and trains the LSTM model.
Model Evaluation: Evaluates the model's performance on the test data.
Visualization: Visualizes the prediction results.
Pre-trained Model (my_model.keras)
The pre-trained LSTM model is saved as my_model.keras. This model is used in the Streamlit application to make predictions. Ensure this file is present in the root directory of the project.

Data Preparation and Model Training
Data Loading: Data is loaded using the yfinance library.
Data Splitting: The data is split into training (70%) and testing (30%) sets.
Scaling: The data is scaled using MinMaxScaler.
Model Training: The LSTM model is trained on the training data.
Model Saving: The trained model is saved for future use.
Example Visualizations
Closing Price vs Time: Plots the closing prices over time.
Closing Price vs Time with 100MA: Plots the closing prices along with the 100-day moving average.
Closing Price vs Time with 100MA & 200MA: Plots the closing prices along with the 100-day and 200-day moving averages.
Predictions vs Original: Compares the predicted prices with the actual prices.
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

License
This project is licensed under the MIT License.# Stock-Trend-Prediction
# Stock-Trend-Prediction
