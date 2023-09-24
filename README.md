# stock_market_project

Stock Market Prediction project built with XGBoost, yfinance, and Streamlit

Web application that allows the user to input a ticker symbol and view relevant stock data

Button to run prediction algorithm which returns the likelihood the stock will increase tomorrow

New column called "Tomorrow" is created by shifting the closing price by 1, revealing tomorrow's stock price for each day. 
By comparing the close and tomorrow columns, new column target is created using 1s and 0s (1 when price goes up, 0 when price goes down)

Uses "Open", "High", "Low", "Volume", "Close" as features
Uses "Target" as the target

Areas to refine: 
Improve the accuracy of the model by tweaking with the XGBoost parameters
Create more in depth features by going through the data