import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from plotly import graph_objs as go
import yfinance as yf

#Returns True if ticker is valid and returns False when ticker is invalid (exception is thrown)
def is_valid_ticker(ticker):
    try:
        data_set = yf.Ticker(ticker)
        _ = data_set.info
        return True
    except Exception as e:
        return False

#From ticker symbol, get the data set from yfinance API and return it
def get_data(ticker):
    #Create the dataset using the ticker input
    data_set = yf.Ticker(ticker)
    data_set = data_set.history(period="max")

    #Remove section that will skew the model
    data_set = data_set.loc["1990-01-01":].copy()
    #Remove unnecessary columns if they exist
    if "Dividends" in data_set.columns:
        del data_set["Dividends"]
    if "Stock Splits" in data_set.columns:
        del data_set["Stock Splits"]
    if "Capital Gains" in data_set.columns:
        del data_set["Capital Gains"]

    return data_set

#Uses the "Date," "Open," and "Close" data from data set to create a custom plotly chart
def plot_data(data_set):
    data_set["Date"] = data_set.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_set["Date"], y=data_set["Open"], name="stock_open", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data_set["Date"], y=data_set["Close"], name="stock_close", line=dict(color='red')))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)

    return fig

#Runs XGBoost ML Algorithm to predict whether tomorrow's stock price increases or decreases
def predict(data_set):
    #Create a "Tomorrow" Close Price column by shifting Close Price data by -1
    data_set["Tomorrow"] = data_set["Close"].shift(-1)

    #Create "Target" column that displays whether the price increased or decreased.
    #1 when price goes up, 0 when price goes down
    data_set["Target"] = (data_set["Tomorrow"] > data_set["Close"]).astype(int)

    #Split the data into training and testing data sets
    train_data = data_set.iloc[:-100]
    test_data = data_set.iloc[-100:]

    #Define the features and the target variable
    #Features(x) will be used to predict the target(y)
    features = trendFeatures(data_set)
    target = "Target"

    X_train = train_data[features]
    y_train = train_data[target]

    X_test = test_data[features]
    y_test = test_data[target]

    #Create and train the model
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    
    #Make and show the predictions on the test data
    y_pred = model.predict(X_test)
    
    #Prediction for tomorrow
    return round(y_pred[-1]*100, 2)

def trendFeatures(data_set):
    #Calculate mean close price from last 2 days, 1 week, 3 months, 1 year, 4 years
    #Provide more information to algorithm so it can make better predictions
    horizons = [2, 5, 60, 250, 1000]

    new_predictors = []

    for horizon in horizons:
        #Calculates the ratio from close price to mean close price over a given period of time
        rolling_averages = data_set.rolling(horizon).mean()
        
        ratio_column = f"Close_Ratio_{horizon}"
        data_set[ratio_column] = data_set["Close"] / rolling_averages["Close"]
        
        #On any given day,looks at the given period of time and sees the number of price increases
        trend_column = f"Trend_{horizon}"
        data_set[trend_column] = data_set.shift(1).rolling(horizon).sum()["Target"]
        
        #These trends (price today compared to price yesterday) is more informative than static numbers
        new_predictors += [ratio_column, trend_column]

    return new_predictors