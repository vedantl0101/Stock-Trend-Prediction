# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import yfinance as yf
# import streamlit as st
# from keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
# import math
# import os

# # Set up Streamlit
# st.set_page_config(page_title="Stock Trend Prediction", layout="wide")
# st.title('Stock Trend Prediction')

# @st.cache_data
# def load_ticker_data():
#     try:
#         # Load ticker data from the provided Excel file
#         file_path = os.path.join(os.getcwd(), 'global_stock_tickers_with_india.xlsx')
#         df = pd.read_excel(file_path)
#         if df.empty:
#             st.error("No ticker data found in the Excel file.")
#             return None
#         return df
#     except Exception as e:
#         st.error(f"Error loading ticker data: {str(e)}")
#         return None

# @st.cache_data
# def load_data(ticker, start, end):
#     try:
#         df = yf.download(ticker, start=start, end=end)
#         if df.empty:
#             st.error(f"No data found for {ticker}. Please check the ticker symbol.")
#             return None
#         return df
#     except Exception as e:
#         st.error(f"Error downloading data: {str(e)}")
#         return None

# # Load the ticker data
# ticker_data = load_ticker_data()
# if ticker_data is None:
#     st.stop()

# # Sector options
# sectors = [
#     "All",
#     "Basic Materials",
#     "Communication Services",
#     "Consumer Cyclical",
#     "Consumer Defensive",
#     "Energy",
#     "Financial Services",
#     "Healthcare",
#     "Industrials",
#     "Real Estate",
#     "Technology",
#     "Utilities"
# ]

# # Taking input for sector selection
# selected_sector = st.selectbox('Select Sector', sectors)

# # Filter tickers based on selected sector
# filtered_ticker_data = ticker_data if selected_sector == "All" else ticker_data[ticker_data['Sector'] == selected_sector]

# # Function to get ticker suggestions based on company name or ticker symbol input
# def get_suggestions(input_text, data):
#     suggestions = data[data['Symbol'].str.contains(input_text, case=False, na=False) | 
#                        data['Name'].str.contains(input_text, case=False, na=False)]
#     return suggestions

# # Taking input for the company name with suggestions
# company_name = st.text_input('Enter Company Name')
# company_suggestions = get_suggestions(company_name, filtered_ticker_data)

# # Display company name and ticker suggestions
# if not company_suggestions.empty:
#     selected_company_name = st.selectbox('Select Company Name', company_suggestions['Name'])
#     ticker = company_suggestions[company_suggestions['Name'] == selected_company_name]['Symbol'].values[0]
# else:
#     ticker = st.text_input('Enter Stock Ticker', 'AAPL')

# # Define the start and end dates
# start = st.date_input('Start date', value=pd.to_datetime('2015-01-01'))
# end = st.date_input('End date', value=pd.to_datetime('2023-12-31'))

# # Validate and download data for the given ticker
# if ticker:
#     df = load_data(ticker, start, end)
#     if df is None:
#         st.stop()

#     # Display data summary and visualizations
#     try:
#         st.subheader('Data Summary')
#         st.write(df.describe())

#         # Closing Price vs Time chart
#         st.subheader('Closing Price vs Time chart')
#         fig = plt.figure(figsize=(12, 6))
#         plt.plot(df.Close, label='Closing Price')
#         plt.xlabel('Date')
#         plt.ylabel('Price')
#         plt.title('Closing Price vs Time')
#         plt.legend()
#         st.pyplot(fig)

#         # Closing Price vs Time chart with 100MA & 200MA
#         st.subheader('Closing Price vs Time chart with 100MA & 200MA')
#         ma100 = df.Close.rolling(100).mean()
#         ma200 = df.Close.rolling(200).mean()
#         fig = plt.figure(figsize=(12, 6))
#         plt.plot(ma100, 'r', label='100MA')
#         plt.plot(ma200, 'g', label='200MA')
#         plt.plot(df.Close, 'b', label='Closing Price')
#         plt.xlabel('Date')
#         plt.ylabel('Price')
#         plt.title('Closing Price with 100MA & 200MA')
#         plt.legend()
#         st.pyplot(fig)

#         # Splitting Data into Training and Testing
#         data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
#         data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])
#         st.write(f"Training data shape: {data_training.shape}")
#         st.write(f"Testing data shape: {data_testing.shape}")

#         # Check if there are enough data points for testing
#         if len(data_testing) < 100:
#             st.error("Insufficient data points available. Please select a larger date range.")
#             st.stop()

#         # Scale the data
#         scaler = MinMaxScaler(feature_range=(0, 1))
#         data_training_array = scaler.fit_transform(data_training)

#         # Load the model
#         try:
#             model_path = os.path.join(os.getcwd(), 'my_model.keras')
#             model = load_model(model_path)
#         except Exception as e:
#             st.error(f"Error loading model: {str(e)}")
#             st.stop()

#         # Prepare the testing data
#         past_100_days = data_training.tail(100)
#         final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
#         input_data = scaler.transform(final_df)

#         x_test = []
#         y_test = []
#         for i in range(100, input_data.shape[0]):
#             x_test.append(input_data[i-100:i])
#             y_test.append(input_data[i, 0])
#         x_test, y_test = np.array(x_test), np.array(y_test)

#         # Make predictions
#         try:
#             with st.spinner('Making predictions...'):
#                 y_predicted = model.predict(x_test)
#         except Exception as e:
#             st.error(f"Error making predictions: {str(e)}")
#             st.stop()

#         # Rescale the predictions and the test data
#         scale_factor = 1 / scaler.scale_[0]
#         y_predicted = y_predicted.flatten() * scale_factor
#         y_test = y_test * scale_factor

#         # Calculate various accuracy metrics
#         mape = mean_absolute_percentage_error(y_test, y_predicted)
#         rmse = math.sqrt(mean_squared_error(y_test, y_predicted))
#         r2 = r2_score(y_test, y_predicted)
        
#         # Calculate directional accuracy
#         direction_test = np.sign(np.diff(y_test))
#         direction_pred = np.sign(np.diff(y_predicted))
#         directional_accuracy = np.mean(direction_test == direction_pred) * 100

#         # Calculate overall accuracy percentage
#         accuracy_percentage = 100 - (mape * 100)

#         st.subheader('Model Performance Metrics')
#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric("MAPE", f"{mape:.2%}")
#             st.write("Mean Absolute Percentage Error (lower is better)")
#             st.metric("R-squared", f"{r2:.4f}")
#             st.write("Coefficient of Determination (higher is better, max 1.0)")
#         with col2:
#             st.metric("RMSE", f"${rmse:.2f}")
#             st.write("Root Mean Square Error (lower is better)")
#             st.metric("Directional Accuracy", f"{directional_accuracy:.2f}%")
#             st.write("Accuracy in predicting price direction")
#             st.metric("Overall Accuracy", f"{accuracy_percentage:.2f}%")
#             st.write("Overall accuracy of the model based on MAPE")

#         # Final Graph
#         st.subheader('Predictions vs Original')
#         fig2 = plt.figure(figsize=(12, 6))
#         plt.plot(y_test, 'b', label='Original Price')
#         plt.plot(y_predicted, 'r', label='Predicted Price')
#         plt.xlabel('Time')
#         plt.ylabel('Price')
#         plt.title('Predictions vs Original Prices')
#         plt.legend()
#         st.pyplot(fig2)

#         # Residual Plot
#         st.subheader('Residual Plot')
#         residuals = y_test - y_predicted
#         fig3 = plt.figure(figsize=(12, 6))
#         plt.scatter(y_predicted, residuals)
#         plt.xlabel('Predicted Values')
#         plt.ylabel('Residuals')
#         plt.title('Residual Plot')
#         plt.axhline(y=0, color='r', linestyle='--')
#         st.pyplot(fig3)

#         # Distribution of Residuals
#         st.subheader('Distribution of Residuals')
#         fig4 = plt.figure(figsize=(12, 6))
#         plt.hist(residuals, bins=50)
#         plt.xlabel('Residuals')
#         plt.ylabel('Frequency')
#         plt.title('Distribution of Residuals')
#         st.pyplot(fig4)

#     except Exception as e:
#         st.error(f"An unexpected error occurred: {str(e)}")
# else:
#     st.error("Please enter a valid company name, ticker symbol, and date range.")


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import yfinance as yf
# import streamlit as st
# from keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
# import math
# import os
# from dotenv import load_dotenv

# # Load environment variables from .env file (if needed for other configurations)
# load_dotenv()

# # Set up Streamlit
# st.set_page_config(page_title="Stock Trend Prediction", layout="wide")
# st.title('Stock Trend Prediction')

# @st.cache_data
# def load_ticker_data():
#     try:
#         # Load ticker data from CSV file
#         file_path = os.path.join(os.getcwd(), 'global_stock_tickers_with_india.csv')
#         df = pd.read_csv(file_path)
#         if df.empty:
#             st.error("No ticker data found in the CSV file.")
#             return None
#         return df
#     except Exception as e:
#         st.error(f"Error loading ticker data: {str(e)}")
#         return None

# @st.cache_data
# def load_data(ticker, start, end):
#     try:
#         df = yf.download(ticker, start=start, end=end)
#         if df.empty:
#             st.error(f"No data found for {ticker}. Please check the ticker symbol.")
#             return None
#         return df
#     except Exception as e:
#         st.error(f"Error downloading data: {str(e)}")
#         return None

# # Load the ticker data
# ticker_data = load_ticker_data()
# if ticker_data is None:
#     st.stop()

# # Sector options
# sectors = [
#     "All",
#     "Basic Materials",
#     "Communication Services",
#     "Consumer Cyclical",
#     "Consumer Defensive",
#     "Energy",
#     "Financial Services",
#     "Healthcare",
#     "Industrials",
#     "Real Estate",
#     "Technology",
#     "Utilities"
# ]

# # Taking input for sector selection
# selected_sector = st.selectbox('Select Sector', sectors)

# # Filter tickers based on selected sector
# filtered_ticker_data = ticker_data if selected_sector == "All" else ticker_data[ticker_data['Sector'] == selected_sector]

# # Taking input for the company name
# company_name = st.text_input('Enter Company Name')

# # Function to get ticker suggestions based on company name or ticker symbol input
# def get_suggestions(input_text, data):
#     suggestions = data[data['Symbol'].str.contains(input_text, case=False, na=False) | 
#                        data['Name'].str.contains(input_text, case=False, na=False)]
#     return suggestions

# company_suggestions = get_suggestions(company_name, filtered_ticker_data)

# # Display company name select box if there are suggestions
# if not company_suggestions.empty:
#     selected_company_name = st.selectbox('Select Company Name', company_suggestions['Name'])
#     ticker = company_suggestions[company_suggestions['Name'] == selected_company_name]['Symbol'].values[0]
    
#     # Add ticker select box
#     ticker_options = company_suggestions['Symbol'].tolist()
#     selected_ticker = st.selectbox('Select Ticker', ticker_options, index=ticker_options.index(ticker))
# else:
#     selected_company_name = None
#     selected_ticker = st.text_input('Enter Ticker Symbol')

# # Define the start and end dates
# start = st.date_input('Start date', value=pd.to_datetime('2015-01-01'))
# end = st.date_input('End date', value=pd.to_datetime('2023-12-31'))

# # Validate ticker
# if selected_ticker:
#     # Download data for the given ticker
#     df = load_data(selected_ticker, start, end)
#     if df is None:
#         st.stop()

#     # Display data summary and visualizations
#     try:
#         st.subheader('Data Summary')
#         st.write(df.describe())

#         # Visualizations
#         st.subheader('Closing Price vs Time chart')
#         fig = plt.figure(figsize=(12, 6))
#         plt.plot(df.Close, label='Closing Price')
#         plt.xlabel('Date')
#         plt.ylabel('Price')
#         plt.title('Closing Price vs Time')
#         plt.legend()
#         st.pyplot(fig)

#         st.subheader('Closing Price vs Time chart with 100MA & 200MA')
#         ma100 = df.Close.rolling(100).mean()
#         ma200 = df.Close.rolling(200).mean()
#         fig = plt.figure(figsize=(12, 6))
#         plt.plot(ma100, 'g', label='100MA')
#         plt.plot(ma200, 'r', label='200MA')
#         plt.plot(df.Close, 'b', label='Closing Price')
#         plt.xlabel('Date')
#         plt.ylabel('Price')
#         plt.title('Closing Price with 100MA & 200MA')
#         plt.legend()
#         st.pyplot(fig)

#         # Splitting Data into Training and Testing
#         data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
#         data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])
#         st.write(f"Training data shape: {data_training.shape}")
#         st.write(f"Testing data shape: {data_testing.shape}")

#         # Check if there are enough data points for testing
#         if len(data_testing) < 100:
#             st.error("Insufficient data points available. Please select a larger date range.")
#             st.stop()

#         # Scale the data
#         scaler = MinMaxScaler(feature_range=(0, 1))
#         data_training_array = scaler.fit_transform(data_training)

#         # Load the model
#         try:
#             model = load_model('my_model.keras')
#         except Exception as e:
#             st.error(f"Error loading model: {str(e)}")
#             st.stop()

#         # Prepare the testing data
#         past_100_days = data_training.tail(100)
#         final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
#         input_data = scaler.transform(final_df)

#         x_test = []
#         y_test = []
#         for i in range(100, input_data.shape[0]):
#             x_test.append(input_data[i-100:i])
#             y_test.append(input_data[i, 0])
#         x_test, y_test = np.array(x_test), np.array(y_test)

#         # Make predictions
#         try:
#             with st.spinner('Making predictions...'):
#                 y_predicted = model.predict(x_test)
#         except Exception as e:
#             st.error(f"Error making predictions: {str(e)}")
#             st.stop()

#         # Rescale the predictions and the test data
#         scale_factor = 1 / scaler.scale_[0]
#         y_predicted = y_predicted.flatten() * scale_factor
#         y_test = y_test * scale_factor

#         # Calculate various accuracy metrics
#         mape = mean_absolute_percentage_error(y_test, y_predicted)
#         rmse = math.sqrt(mean_squared_error(y_test, y_predicted))
#         r2 = r2_score(y_test, y_predicted)
        
#         # Calculate directional accuracy
#         direction_test = np.sign(np.diff(y_test))
#         direction_pred = np.sign(np.diff(y_predicted))
#         directional_accuracy = np.mean(direction_test == direction_pred) * 100

#         # Calculate overall accuracy percentage
#         accuracy_percentage = 100 - (mape * 100)

#         st.subheader('Model Performance Metrics')
#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric("MAPE", f"{mape:.2%}")
#             st.write("Mean Absolute Percentage Error (lower is better)")
#             st.metric("R-squared", f"{r2:.4f}")
#             st.write("Coefficient of Determination (higher is better, max 1.0)")
#         with col2:
#             st.metric("RMSE", f"{rmse:.2f}")
#             st.write("Root Mean Square Error (lower is better)")
#             st.metric("Directional Accuracy", f"{directional_accuracy:.2f}%")
#             st.write("Accuracy in predicting price direction")
#             st.metric("Overall Accuracy", f"{accuracy_percentage:.2f}%")
#             st.write("Overall accuracy of the model based on MAPE")

#         # Final Graph
#         st.subheader('Predictions vs Original')
#         fig2 = plt.figure(figsize=(12, 6))
#         plt.plot(y_test, 'b', label='Original Price')
#         plt.plot(y_predicted, 'r', label='Predicted Price')
#         plt.xlabel('Time')
#         plt.ylabel('Price')
#         plt.title('Predictions vs Original Prices')
#         plt.legend()
#         st.pyplot(fig2)

#         # Residual Plot
#         st.subheader('Residual Plot')
#         residuals = y_test - y_predicted
#         fig3 = plt.figure(figsize=(12, 6))
#         plt.scatter(y_predicted, residuals)
#         plt.xlabel('Predicted Values')
#         plt.ylabel('Residuals')
#         plt.title('Residual Plot')
#         plt.axhline(y=0, color='r', linestyle='--')
#         st.pyplot(fig3)

#         # Distribution of Residuals
#         st.subheader('Distribution of Residuals')
#         fig4 = plt.figure(figsize=(12, 6))
#         plt.hist(residuals, bins=50)
#         plt.xlabel('Residuals')
#         plt.ylabel('Frequency')
#         plt.title('Distribution of Residuals')
#         st.pyplot(fig4)

#     except Exception as e:
#         st.error(f"An unexpected error occurred: {str(e)}")
# else:
#     st.error("Please enter a valid company name, ticker symbol and date range.")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import math
import os
from dotenv import load_dotenv

# Load environment variables from .env file (if needed for other configurations)
load_dotenv()

# Set up Streamlit
st.set_page_config(page_title="Stock Trend Prediction", layout="wide")
st.title('Stock Trend Prediction')

# Use Streamlit's caching mechanism effectively to load data
@st.cache_data(show_spinner=False)
def load_ticker_data():
    try:
        file_path = os.path.join(os.getcwd(), 'global_stock_tickers_with_india.csv')
        df = pd.read_csv(file_path)
        if df.empty:
            st.error("No ticker data found in the CSV file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading ticker data: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            st.error(f"No data found for {ticker}. Please check the ticker symbol.")
            return None
        return df
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None

# Load the ticker data
ticker_data = load_ticker_data()
if ticker_data is None:
    st.stop()

# Sector options
sectors = [
    "All",
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities"
]

# Taking input for sector selection
selected_sector = st.selectbox('Select Sector', sectors)

# Filter tickers based on selected sector
filtered_ticker_data = ticker_data if selected_sector == "All" else ticker_data[ticker_data['Sector'] == selected_sector]

# Taking input for the company name
company_name = st.text_input('Enter Company Name')

# Function to get ticker suggestions based on company name or ticker symbol input
def get_suggestions(input_text, data):
    suggestions = data[data['Symbol'].str.contains(input_text, case=False, na=False) | 
                       data['Name'].str.contains(input_text, case=False, na=False)]
    return suggestions

company_suggestions = get_suggestions(company_name, filtered_ticker_data)

# Display company name select box if there are suggestions
if not company_suggestions.empty:
    selected_company_name = st.selectbox('Select Company Name', company_suggestions['Name'])
    ticker = company_suggestions[company_suggestions['Name'] == selected_company_name]['Symbol'].values[0]
    
    # Add ticker select box
    ticker_options = company_suggestions['Symbol'].tolist()
    selected_ticker = st.selectbox('Select Ticker', ticker_options, index=ticker_options.index(ticker))
else:
    selected_company_name = None
    selected_ticker = st.text_input('Enter Ticker Symbol')

# Define the start and end dates
start = st.date_input('Start date', value=pd.to_datetime('2015-01-01'))
end = st.date_input('End date', value=pd.to_datetime('2023-12-31'))

# Validate ticker
if selected_ticker:
    # Download data for the given ticker
    df = load_data(selected_ticker, start, end)
    if df is None:
        st.stop()

    # Display data summary and visualizations
    try:
        st.subheader('Data Summary')
        st.write(df.describe())

        # Visualizations
        st.subheader('Closing Price vs Time chart')
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df.Close, label='Closing Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Closing Price vs Time')
        plt.legend()
        st.pyplot(fig)

        st.subheader('Closing Price vs Time chart with 100MA & 200MA')
        ma100 = df.Close.rolling(100).mean()
        ma200 = df.Close.rolling(200).mean()
        fig = plt.figure(figsize=(12, 6))
        plt.plot(ma100, 'g', label='100MA')
        plt.plot(ma200, 'r', label='200MA')
        plt.plot(df.Close, 'b', label='Closing Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Closing Price with 100MA & 200MA')
        plt.legend()
        st.pyplot(fig)

        # # Splitting Data into Training and Testing
        # data_training = df['Close'][:int(len(df) * 0.70)]
        # data_testing = df['Close'][int(len(df) * 0.70):]
        # st.write(f"Training data shape: {data_training.shape}")
        # st.write(f"Testing data shape: {data_testing.shape}")

        # Splitting Data into Training and Testing
        data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])
        st.write(f"Training data shape: {data_training.shape}")
        st.write(f"Testing data shape: {data_testing.shape}")

        
        # Check if there are enough data points for testing
        if len(data_testing) < 100:
            st.error("Insufficient data points available. Please select a larger date range.")
            st.stop()

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(np.array(data_training).reshape(-1, 1))

        # Load the model
        @st.cache_data(show_spinner=False, persist=True)
        def load_model_keras():
            try:
                return load_model('my_model.keras')
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return None

        model = load_model_keras()
        if model is None:
            st.stop()

        # Prepare the testing data
        past_100_days = data_training[-100:]
        final_df = pd.concat([past_100_days, data_testing])
        input_data = scaler.transform(np.array(final_df).reshape(-1, 1))

        x_test = []
        y_test = []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Make predictions
        try:
            with st.spinner('Making predictions...'):
                y_predicted = model.predict(x_test)
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            st.stop()

        # Rescale the predictions and the test data
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted.flatten() * scale_factor
        y_test = y_test * scale_factor

        # Calculate various accuracy metrics
        mape = mean_absolute_percentage_error(y_test, y_predicted)
        rmse = math.sqrt(mean_squared_error(y_test, y_predicted))
        r2 = r2_score(y_test, y_predicted)
        
        # Calculate directional accuracy
        direction_test = np.sign(np.diff(y_test))
        direction_pred = np.sign(np.diff(y_predicted))
        directional_accuracy = np.mean(direction_test == direction_pred) * 100

        # Calculate overall accuracy percentage
        accuracy_percentage = 100 - (mape * 100)

        st.subheader('Model Performance Metrics')
        col1, col2 = st.columns(2)
        with col1:
            st.metric("MAPE", f"{mape:.2%}")
            st.write("Mean Absolute Percentage Error (lower is better)")
            st.metric("R-squared", f"{r2:.4f}")
            st.write("Coefficient of Determination (higher is better, max 1.0)")
        with col2:
            st.metric("RMSE", f"{rmse:.2f}")
            st.write("Root Mean Square Error (lower is better)")
            st.metric("Directional Accuracy", f"{directional_accuracy:.2f}%")
            st.write("Accuracy in predicting price direction")
            st.metric("Overall Accuracy", f"{accuracy_percentage:.2f}%")
            st.write("Overall accuracy of the model based on MAPE")

        # Final Graph
        st.subheader('Predictions vs Original')
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(y_test, 'b', label='Original Price')
        plt.plot(y_predicted, 'r', label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Predictions vs Original Prices')
        plt.legend()
        st.pyplot(fig2)

        # Residual Plot
        st.subheader('Residual Plot')
        residuals = y_test - y_predicted
        fig3 = plt.figure(figsize=(12, 6))
        plt.scatter(y_predicted, residuals)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.axhline(y=0, color='r', linestyle='--')
        st.pyplot(fig3)

        # Distribution of Residuals
        st.subheader('Distribution of Residuals')
        fig4 = plt.figure(figsize=(12, 6))
        plt.hist(residuals, bins=50)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        st.pyplot(fig4)

    except Exception as e:
        st.error(f"An error occurred while processing data: {str(e)}")
else:
    st.warning("Please enter a valid company name, ticker symbol and date range.")


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import yfinance as yf
# import streamlit as st
# from keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
# import math
# import os
# from dotenv import load_dotenv

# # Load environment variables from .env file (if needed for other configurations)
# load_dotenv()

# # Set up Streamlit
# st.set_page_config(page_title="Stock Trend Prediction", layout="wide")
# st.title('Stock Trend Prediction')

# # Use Streamlit's caching mechanism effectively to load data
# @st.cache_data(show_spinner=False)
# def load_ticker_data():
#     try:
#         file_path = os.path.join(os.getcwd(), 'global_stock_tickers_with_india.csv')
#         df = pd.read_csv(file_path)
#         if df.empty:
#             st.error("No ticker data found in the CSV file.")
#             return None
#         return df
#     except Exception as e:
#         st.error(f"Error loading ticker data: {str(e)}")
#         return None

# @st.cache_data(show_spinner=False)
# def load_data(ticker, start, end):
#     try:
#         df = yf.download(ticker, start=start, end=end)
#         if df.empty:
#             st.error(f"No data found for {ticker}. Please check the ticker symbol.")
#             return None
#         return df
#     except Exception as e:
#         st.error(f"Error downloading data: {str(e)}")
#         return None

# # Load the ticker data
# ticker_data = load_ticker_data()
# if ticker_data is None:
#     st.stop()

# # Sector options
# sectors = [
#     "All",
#     "Basic Materials",
#     "Communication Services",
#     "Consumer Cyclical",
#     "Consumer Defensive",
#     "Energy",
#     "Financial Services",
#     "Healthcare",
#     "Industrials",
#     "Real Estate",
#     "Technology",
#     "Utilities"
# ]

# # Taking input for sector selection
# selected_sector = st.selectbox('Select Sector', sectors)

# # Filter tickers based on selected sector
# filtered_ticker_data = ticker_data if selected_sector == "All" else ticker_data[ticker_data['Sector'] == selected_sector]

# # Taking input for the company name
# company_name = st.text_input('Enter Company Name')

# # Function to get ticker suggestions based on company name or ticker symbol input
# def get_suggestions(input_text, data):
#     suggestions = data[data['Symbol'].str.contains(input_text, case=False, na=False) | 
#                        data['Name'].str.contains(input_text, case=False, na=False)]
#     return suggestions

# company_suggestions = get_suggestions(company_name, filtered_ticker_data)

# # Display company name select box if there are suggestions
# if not company_suggestions.empty:
#     selected_company_name = st.selectbox('Select Company Name', company_suggestions['Name'])
#     ticker = company_suggestions[company_suggestions['Name'] == selected_company_name]['Symbol'].values[0]
    
#     # Add ticker select box
#     ticker_options = company_suggestions['Symbol'].tolist()
#     selected_ticker = st.selectbox('Select Ticker', ticker_options, index=ticker_options.index(ticker))
# else:
#     selected_company_name = None
#     selected_ticker = st.text_input('Enter Ticker Symbol')

# # Define the start and end dates
# start = st.date_input('Start date', value=pd.to_datetime('2015-01-01'))
# end = st.date_input('End date', value=pd.to_datetime('2023-12-31'))

# # Validate ticker
# if selected_ticker:
#     # Download data for the given ticker
#     df = load_data(selected_ticker, start, end)
#     if df is None:
#         st.stop()

#     # Display data summary and visualizations
#     try:
#         st.subheader('Data Summary')
#         st.write(df.describe())

#         # Visualizations
#         st.subheader('Closing Price vs Time chart')
#         fig = plt.figure(figsize=(12, 6))
#         plt.plot(df.Close, label='Closing Price')
#         plt.xlabel('Date')
#         plt.ylabel('Price')
#         plt.title('Closing Price vs Time')
#         plt.legend()
#         st.pyplot(fig)

#         st.subheader('Closing Price vs Time chart with 100MA & 200MA')
#         ma100 = df.Close.rolling(100).mean()
#         ma200 = df.Close.rolling(200).mean()
#         fig = plt.figure(figsize=(12, 6))
#         plt.plot(ma100, 'g', label='100MA')
#         plt.plot(ma200, 'r', label='200MA')
#         plt.plot(df.Close, 'b', label='Closing Price')
#         plt.xlabel('Date')
#         plt.ylabel('Price')
#         plt.title('Closing Price with 100MA & 200MA')
#         plt.legend()
#         st.pyplot(fig)

#         # Splitting Data into Training and Testing
#         # data_training = df['Close'][:int(len(df) * 0.70)]
#         # data_testing = df['Close'][int(len(df) * 0.70):]
#         # st.write(f"Training data shape: {data_training.shape}")
#         # st.write(f"Testing data shape: {data_testing.shape}")
#         data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
#         data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])
#         st.write(f"Training data shape: {data_training.shape}")
#         st.write(f"Testing data shape: {data_testing.shape}")

#         # Check if there are enough data points for testing
#         if len(data_testing) < 100:
#             st.error("Insufficient data points available. Please select a larger date range.")
#             st.stop()

#         # Scale the data
#         scaler = MinMaxScaler(feature_range=(0, 1))
#         data_training_array = scaler.fit_transform(np.array(data_training).reshape(-1, 1))

#         # Load the model
#         @st.cache_data(show_spinner=False, persist=True)
#         def load_model_keras():
#             try:
#                 return load_model('my_model.keras')
#             except Exception as e:
#                 st.error(f"Error loading model: {str(e)}")
#                 return None

#         model = load_model_keras()
#         if model is None:
#             st.stop()

#         # Prepare the testing data
#         past_100_days = data_training[-100:]
#         final_df = pd.concat([past_100_days, data_testing])
#         input_data = scaler.transform(np.array(final_df).reshape(-1, 1))

#         x_test = []
#         y_test = []
#         for i in range(100, input_data.shape[0]):
#             x_test.append(input_data[i-100:i])
#             y_test.append(input_data[i, 0])
#         x_test, y_test = np.array(x_test), np.array(y_test)

#         # Make predictions
#         try:
#             with st.spinner('Making predictions...'):
#                 y_predicted = model.predict(x_test)
#         except Exception as e:
#             st.error(f"Error making predictions: {str(e)}")
#             st.stop()

#         # Rescale the predictions and the test data
#         scale_factor = 1 / scaler.scale_[0]
#         y_predicted = y_predicted.flatten() * scale_factor
#         y_test = y_test * scale_factor

#         # Calculate various accuracy metrics
#         mape = mean_absolute_percentage_error(y_test, y_predicted)
#         rmse = math.sqrt(mean_squared_error(y_test, y_predicted))
#         r2 = r2_score(y_test, y_predicted)
        
#         # Calculate directional accuracy
#         direction_test = np.sign(np.diff(y_test))
#         direction_pred = np.sign(np.diff(y_predicted))
#         directional_accuracy = np.mean(direction_test == direction_pred) * 100

#         # Calculate overall accuracy percentage
#         accuracy_percentage = 100 - (mape * 100)

#         st.subheader('Model Performance Metrics')
#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric("MAPE", f"{mape:.2%}")
#             st.write("Mean Absolute Percentage Error (lower is better)")
#             st.metric("R-squared", f"{r2:.4f}")
#             st.write("Coefficient of Determination (higher is better, max 1.0)")
#         with col2:
#             st.metric("RMSE", f"{rmse:.2f}")
#             st.write("Root Mean Square Error (lower is better)")
#             st.metric("Directional Accuracy", f"{directional_accuracy:.2f}%")
#             st.write("Accuracy in predicting price direction")
#             st.metric("Overall Accuracy", f"{accuracy_percentage:.2f}%")
#             st.write("Overall accuracy of the model based on MAPE")

#         # Final Graph
#         st.subheader('Predictions vs Original')
#         fig2 = plt.figure(figsize=(12, 6))
#         plt.plot(y_test, 'b', label='Original Price')
#         plt.plot(y_predicted, 'r', label='Predicted Price')
#         plt.xlabel('Time')
#         plt.ylabel('Price')
#         plt.title('Predictions vs Original Prices')
#         plt.legend()
#         st.pyplot(fig2)

#         # Residual Plot
#         st.subheader('Residual Plot')
#         residuals = y_test - y_predicted
#         fig3 = plt.figure(figsize=(12, 6))
#         plt.scatter(y_predicted, residuals)
#         plt.xlabel('Predicted Values')
#         plt.ylabel('Residuals')
#         plt.title('Residual Plot')
#         plt.axhline(y=0, color='r', linestyle='--')
#         st.pyplot(fig3)

#         # Distribution of Residuals
#         st.subheader('Distribution of Residuals')
#         fig4 = plt.figure(figsize=(12, 6))
#         plt.hist(residuals, bins=50)
#         plt.xlabel('Residuals')
#         plt.ylabel('Frequency')
#         plt.title('Distribution of Residuals')
#         st.pyplot(fig4)

#     except Exception as e:
#         st.error(f"An error occurred while processing data: {str(e)}")
# else:
#     st.warning("Please enter a valid company name, ticker symbol and date range.")

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import yfinance as yf
# import streamlit as st
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
# import math
# import os
# from dotenv import load_dotenv
# import tensorflow as tf

# # Load environment variables from .env file (if needed for other configurations)
# load_dotenv()

# # Set up Streamlit
# st.set_page_config(page_title="Stock Trend Prediction", layout="wide")
# st.title('Stock Trend Prediction')

# # Use Streamlit's caching mechanism effectively to load data
# @st.cache_data(show_spinner=False)
# def load_ticker_data():
#     try:
#         file_path = os.path.join(os.getcwd(), 'global_stock_tickers_with_india.csv')
#         df = pd.read_csv(file_path)
#         if df.empty:
#             st.error("No ticker data found in the CSV file.")
#             return None
#         return df
#     except Exception as e:
#         st.error(f"Error loading ticker data: {str(e)}")
#         return None

# @st.cache_data(show_spinner=False)
# def load_data(ticker, start, end):
#     try:
#         df = yf.download(ticker, start=start, end=end)
#         if df.empty:
#             st.error(f"No data found for {ticker}. Please check the ticker symbol.")
#             return None
#         return df
#     except Exception as e:
#         st.error(f"Error downloading data: {str(e)}")
#         return None

# # Load the ticker data
# ticker_data = load_ticker_data()
# if ticker_data is None:
#     st.stop()

# # Sector options
# sectors = [
#     "All",
#     "Basic Materials",
#     "Communication Services",
#     "Consumer Cyclical",
#     "Consumer Defensive",
#     "Energy",
#     "Financial Services",
#     "Healthcare",
#     "Industrials",
#     "Real Estate",
#     "Technology",
#     "Utilities"
# ]

# # Taking input for sector selection
# selected_sector = st.selectbox('Select Sector', sectors)

# # Filter tickers based on selected sector
# filtered_ticker_data = ticker_data if selected_sector == "All" else ticker_data[ticker_data['Sector'] == selected_sector]

# # Taking input for the company name
# company_name = st.text_input('Enter Company Name')

# # Function to get ticker suggestions based on company name or ticker symbol input
# def get_suggestions(input_text, data):
#     suggestions = data[data['Symbol'].str.contains(input_text, case=False, na=False) | 
#                        data['Name'].str.contains(input_text, case=False, na=False)]
#     return suggestions

# company_suggestions = get_suggestions(company_name, filtered_ticker_data)

# # Display company name select box if there are suggestions
# if not company_suggestions.empty:
#     selected_company_name = st.selectbox('Select Company Name', company_suggestions['Name'])
#     ticker = company_suggestions[company_suggestions['Name'] == selected_company_name]['Symbol'].values[0]
    
#     # Add ticker select box
#     ticker_options = company_suggestions['Symbol'].tolist()
#     selected_ticker = st.selectbox('Select Ticker', ticker_options, index=ticker_options.index(ticker))
# else:
#     selected_company_name = None
#     selected_ticker = st.text_input('Enter Ticker Symbol')

# # Define the start and end dates
# start = st.date_input('Start date', value=pd.to_datetime('2015-01-01'))
# end = st.date_input('End date', value=pd.to_datetime('2023-12-31'))

# # Validate ticker
# if selected_ticker:
#     # Download data for the given ticker
#     df = load_data(selected_ticker, start, end)
#     if df is None:
#         st.stop()

#     # Display data summary and visualizations
#     try:
#         st.subheader('Data Summary')
#         st.write(df.describe())

#         # Visualizations
#         st.subheader('Closing Price vs Time chart')
#         fig = plt.figure(figsize=(12, 6))
#         plt.plot(df.Close, label='Closing Price')
#         plt.xlabel('Date')
#         plt.ylabel('Price')
#         plt.title('Closing Price vs Time')
#         plt.legend()
#         st.pyplot(fig)

#         st.subheader('Closing Price vs Time chart with 100MA & 200MA')
#         ma100 = df.Close.rolling(100).mean()
#         ma200 = df.Close.rolling(200).mean()
#         fig = plt.figure(figsize=(12, 6))
#         plt.plot(ma100, 'g', label='100MA')
#         plt.plot(ma200, 'r', label='200MA')
#         plt.plot(df.Close, 'b', label='Closing Price')
#         plt.xlabel('Date')
#         plt.ylabel('Price')
#         plt.title('Closing Price with 100MA & 200MA')
#         plt.legend()
#         st.pyplot(fig)

#         # Splitting Data into Training and Testing
#         data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
#         data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])
#         st.write(f"Training data shape: {data_training.shape}")
#         st.write(f"Testing data shape: {data_testing.shape}")

#         # Check if there are enough data points for testing
#         if len(data_testing) < 100:
#             st.error("Insufficient data points available. Please select a larger date range.")
#             st.stop()

#         # Scale the data
#         scaler = MinMaxScaler(feature_range=(0, 1))
#         data_training_array = scaler.fit_transform(np.array(data_training).reshape(-1, 1))

#         # Load the model
#         model = load_model_keras()
#         if model is None:
#             st.stop()

#         # Prepare the testing data
#         past_100_days = data_training_array[-100:]
#         final_df = pd.concat([pd.DataFrame(past_100_days), data_testing], ignore_index=True)
#         input_data = scaler.transform(np.array(final_df).reshape(-1, 1))

#         x_test = []
#         y_test = []
#         for i in range(100, input_data.shape[0]):
#             x_test.append(input_data[i-100:i])
#             y_test.append(input_data[i, 0])
#         x_test, y_test = np.array(x_test), np.array(y_test)

#         # Make predictions
#         try:
#             with st.spinner('Making predictions...'):
#                 y_predicted = model.predict(x_test)
#         except Exception as e:
#             st.error(f"Error making predictions: {str(e)}")
#             st.stop()

#         # Rescale the predictions and the test data
#         scale_factor = 1 / scaler.scale_[0]
#         y_predicted = y_predicted.flatten() * scale_factor
#         y_test = y_test * scale_factor

#         # Calculate various accuracy metrics
#         mape = mean_absolute_percentage_error(y_test, y_predicted)
#         rmse = math.sqrt(mean_squared_error(y_test, y_predicted))
#         r2 = r2_score(y_test, y_predicted)
        
#         # Calculate directional accuracy
#         direction_test = np.sign(np.diff(y_test))
#         direction_pred = np.sign(np.diff(y_predicted))
#         directional_accuracy = np.mean(direction_test == direction_pred) * 100

#         # Display accuracy metrics
#         st.subheader('Accuracy Metrics')
#         st.write(f"Mean Absolute Percentage Error (MAPE): {mape}")
#         st.write(f"Root Mean Squared Error (RMSE): {rmse}")
#         st.write(f"R-squared: {r2}")
#         st.write(f"Directional Accuracy: {directional_accuracy}%")

#         # Visualize predictions vs original prices
#         st.subheader('Predictions vs Original')
#         fig2 = plt.figure(figsize=(12, 6))
#         plt.plot(y_test, 'b', label='Original Price')
#         plt.plot(y_predicted, 'r', label='Predicted Price')
#         plt.xlabel('Time')
#         plt.ylabel('Price')
#         plt.title('Predictions vs Original')
#         plt.legend()
#         st.pyplot(fig2)

#         # Residual Plot
#         residuals = y_test - y_predicted
#         st.subheader('Residual Plot')
#         fig3 = plt.figure(figsize=(12, 6))
#         plt.plot(residuals)
#         plt.xlabel('Time')
#         plt.ylabel('Residuals')
#         plt.title('Residual Plot')
#         plt.axhline(y=0, color='r', linestyle='--')
#         st.pyplot(fig3)

#         # Distribution of Residuals
#         st.subheader('Distribution of Residuals')
#         fig4 = plt.figure(figsize=(12, 6))
#         plt.hist(residuals, bins=50)
#         plt.xlabel('Residuals')
#         plt.ylabel('Frequency')
#         plt.title('Distribution of Residuals')
#         st.pyplot(fig4)

#     except Exception as e:
#         st.error(f"An error occurred while processing data: {str(e)}")
# else:
#     st.warning("Please enter a valid company name, ticker symbol and date range.")



