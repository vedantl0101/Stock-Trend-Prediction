# Stock Trend Prediction

This project uses a Long Short-Term Memory (LSTM) neural network to predict stock prices based on historical data. The project is implemented using Python, leveraging libraries such as TensorFlow/Keras for building the model, Streamlit for the web interface, and yfinance for fetching stock data.

## Project Structure
- **app.py:** Main application file that sets up the Streamlit interface, fetches data, preprocesses it, and displays the prediction results.
- **my_model.keras:** Pre-trained LSTM model used for making predictions.
- **Stock Trend Prediction.ipynb:** Jupyter Notebook containing the code for data processing, model training, and evaluation.

## Features
- **Data Fetching:** Fetches historical stock data using the yfinance library.
- **Data Visualization:** Visualizes the stock's closing prices along with 100-day and 200-day moving averages.
- **Model Training:** Trains an LSTM model to predict stock prices based on past data.
- **Prediction:** Uses the trained model to predict future stock prices and compares them with actual prices.

## Installation
Clone the repository:

```bash
git clone https://github.com/vedantl0101/stock-trend-prediction.git
cd stock-trend-prediction

