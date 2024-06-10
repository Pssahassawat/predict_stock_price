import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model 
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

model = load_model('./lstm_model.h5')

start = '2010-01-01'
# end = 'none'

st.title('Stock Price Prediction')
user_input = st.text_input('Enter a stock symbol')


scaler = MinMaxScaler(feature_range=(0,1))

if (len(user_input) == 0):
    st.text('Please enter symbol')
else:
    data = yf.download(user_input, start=start)
    data_training = data[['Open', 'Close']][0:int(len(data)*0.70)]
    data_testing = data[['Open', 'Close']][int(len(data)*0.70):int(len(data))]
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)
 
    x_test = []
    y_test = []

    for i in range(100 , input_data.shape[0]):
        x_test.append(input_data[i - 100: i])
        y_test.append(input_data[i,0])
        
    x_test, y_test = np.array(x_test), np.array(y_test) 
    y_predicted =  model.predict(x_test)
    min = scaler.data_min_
    range = scaler.data_range_
    y_predicted = (y_predicted * range[1]) + min[1]
    y_test = (y_test * range[1]) + min[1]
    df_final = final_df['Close'][100:]
    df_final = df_final.reset_index()
    df_final = df_final['Close']
    ma100 =df_final.rolling(window=100).mean()
    ma200 =df_final.rolling(window=200).mean()
    plt.figure(figsize=(12,6))
    plt.plot(df_final, 'b', label = 'Original Price')
    plt.plot(y_predicted, 'r', label = 'Predicted Price')
    plt.plot(ma100, 'g', label = 'MA100')
    plt.plot(ma200, 'y', label = 'MA200')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt.show())
    st.subheader('Compare')
