!pip install matplotlib
!pip install yfinance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

start = '2010-01-01'
# end = 'none'

st.title('Stock Price Prediction')
user_input = st.text_input('Enter a stock symbol')
data = yf.download(user_input,start=start)

st.subheader('Data from 2010 to now')
st.write(data.describe())
