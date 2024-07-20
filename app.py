import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import math

st.set_page_config(page_title="Stock Trend Prediction", layout="wide")
st.title('Stock Trend Prediction')

@st.cache_data
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

# Taking input for the stock ticker
ticker = st.text_input('Enter Stock Ticker', 'AAPL')

# Define the start and end dates
start = st.date_input('Start date', value=pd.to_datetime('2015-01-01'))
end = st.date_input('End date', value=pd.to_datetime('2023-12-31'))

# Download data for the given ticker
df = load_data(ticker, start, end)

if df is not None:
    st.subheader('Data Summary')
    st.write(df.describe())

    # Visualizations
    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, 'r', label='100MA')
    plt.plot(ma200, 'g', label='200MA')
    plt.plot(df.Close, 'b', label='Closing Price')
    plt.legend()
    st.pyplot(fig)

    # Splitting Data into Training and Testing
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])
    st.write(f"Training data shape: {data_training.shape}")
    st.write(f"Testing data shape: {data_testing.shape}")

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    # Load the model
    try:
        model = load_model('my_model.keras')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

    # Prepare the testing data
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.transform(final_df)

    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)

    # Make predictions
    with st.spinner('Making predictions...'):
        y_predicted = model.predict(x_test)

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
        st.metric("RMSE", f"${rmse:.2f}")
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
    plt.legend()
    st.pyplot(fig2)

    # Residual Plot
    st.subheader('Residual Plot')
    residuals = y_test - y_predicted
    fig3 = plt.figure(figsize=(12, 6))
    plt.scatter(y_predicted, residuals)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='--')
    st.pyplot(fig3)

    # Distribution of Residuals
    st.subheader('Distribution of Residuals')
    fig4 = plt.figure(figsize=(12, 6))
    plt.hist(residuals, bins=50)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    st.pyplot(fig4)

else:
    st.write("Please enter a valid stock ticker and date range.")















# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import yfinance as yf
# import streamlit as st
# from keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler

# st.title('Stock Trend Prediction')

# # Taking input for the stock ticker
# tickers = st.text_input('Enter Stock Ticker', 'AAPL')

# # Define the start and end dates
# start = '2015-01-01'
# end = '2023-12-31'

# # Download data for the given ticker
# df = yf.download(tickers, start=start, end=end)

# st.subheader('Data from 2015 - 2023')
# st.write(df.describe())

# # Visualizations
# st.subheader('Closing Price vs Time chart')
# fig = plt.figure(figsize=(12, 6))
# plt.plot(df.Close)
# st.pyplot(fig)

# st.subheader('Closing Price vs Time chart with 100MA')
# ma100 = df.Close.rolling(100).mean()
# fig = plt.figure(figsize=(12, 6))
# plt.plot(ma100, label='100MA')
# plt.plot(df.Close, label='Closing Price')
# plt.legend()
# st.pyplot(fig)

# st.subheader('Closing Price vs Time chart with 100MA & 200MA')
# ma100 = df.Close.rolling(100).mean()
# ma200 = df.Close.rolling(200).mean()
# fig = plt.figure(figsize=(12, 6))
# plt.plot(ma100, 'r', label='100MA')
# plt.plot(ma200, 'g', label='200MA')
# plt.plot(df.Close, 'b', label='Closing Price')
# plt.legend()
# st.pyplot(fig)

# # Splitting Data into Training and Testing
# data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
# data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])
# st.write(f"Training data shape: {data_training.shape}")
# st.write(f"Testing data shape: {data_testing.shape}")

# # Scale the data
# scaler = MinMaxScaler(feature_range=(0, 1))
# data_training_array = scaler.fit_transform(data_training)

# # Load the model
# model = load_model('my_model.keras')

# # Prepare the testing data
# past_100_days = data_training.tail(100)
# final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
# input_data = scaler.fit_transform(final_df)

# x_test = []
# y_test = []
# for i in range(100, input_data.shape[0]):
#     x_test.append(input_data[i-100:i])
#     y_test.append(input_data[i, 0])
# x_test, y_test = np.array(x_test), np.array(y_test)

# # Make predictions
# y_predicted = model.predict(x_test)

# # Rescale the predictions and the test data
# scale_factor = 1 / scaler.scale_[0]
# y_predicted = y_predicted * scale_factor
# y_test = y_test * scale_factor

# # Final Graph
# st.subheader('Predictions vs Original')
# fig2 = plt.figure(figsize=(12, 6))
# plt.plot(y_test, 'b', label='Original Price')
# plt.plot(y_predicted, 'r', label='Predicted Price')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# st.pyplot(fig2)

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import yfinance as yf
# import streamlit as st
# from keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_percentage_error

# st.set_page_config(page_title="Stock Trend Prediction", layout="wide")
# st.title('Stock Trend Prediction')

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

# # Taking input for the stock ticker
# ticker = st.text_input('Enter Stock Ticker', 'AAPL')

# # Define the start and end dates
# start = st.date_input('Start date', value=pd.to_datetime('2015-01-01'))
# end = st.date_input('End date', value=pd.to_datetime('2023-12-31'))

# # Download data for the given ticker
# df = load_data(ticker, start, end)

# if df is not None:
#     st.subheader('Data Summary')
#     st.write(df.describe())

#     # Visualizations
#     st.subheader('Closing Price vs Time chart')
#     fig = plt.figure(figsize=(12, 6))
#     plt.plot(df.Close)
#     st.pyplot(fig)

#     st.subheader('Closing Price vs Time chart with 100MA & 200MA')
#     ma100 = df.Close.rolling(100).mean()
#     ma200 = df.Close.rolling(200).mean()
#     fig = plt.figure(figsize=(12, 6))
#     plt.plot(ma100, 'r', label='100MA')
#     plt.plot(ma200, 'g', label='200MA')
#     plt.plot(df.Close, 'b', label='Closing Price')
#     plt.legend()
#     st.pyplot(fig)

#     # Splitting Data into Training and Testing
#     data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
#     data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])
#     st.write(f"Training data shape: {data_training.shape}")
#     st.write(f"Testing data shape: {data_testing.shape}")

#     # Scale the data
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     data_training_array = scaler.fit_transform(data_training)

#     # Load the model
#     try:
#         model = load_model('my_model.keras')
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         st.stop()

#     # Prepare the testing data
#     past_100_days = data_training.tail(100)
#     final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
#     input_data = scaler.transform(final_df)

#     x_test = []
#     y_test = []
#     for i in range(100, input_data.shape[0]):
#         x_test.append(input_data[i-100:i])
#         y_test.append(input_data[i, 0])
#     x_test, y_test = np.array(x_test), np.array(y_test)

#     # Make predictions
#     with st.spinner('Making predictions...'):
#         y_predicted = model.predict(x_test)

#     # Rescale the predictions and the test data
#     scale_factor = 1 / scaler.scale_[0]
#     y_predicted = y_predicted * scale_factor
#     y_test = y_test * scale_factor

#     # Calculate MAPE
#     mape = mean_absolute_percentage_error(y_test, y_predicted)
#     accuracy = 100 - mape

#     st.subheader('Model Accuracy')
#     st.write(f"The model's accuracy is {accuracy:.2f}%")
#     st.write("(Based on Mean Absolute Percentage Error)")

#     # Final Graph
#     st.subheader('Predictions vs Original')
#     fig2 = plt.figure(figsize=(12, 6))
#     plt.plot(y_test, 'b', label='Original Price')
#     plt.plot(y_predicted, 'r', label='Predicted Price')
#     plt.xlabel('Time')
#     plt.ylabel('Price')
#     plt.legend()
#     st.pyplot(fig2)
# else:
#     st.write("Please enter a valid stock ticker and date range.")




