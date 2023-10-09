## Stock Market Prediction Project
Personal project built with:
* XGBoost: ML Decision Tree algorithm used for predicting tomorrow's stock data
* sklearn: Random Forest Classifier (ML Decision Tree algorithm) used for prediction
* yfinance: For retrieving up-to-date stock market data sets
* Streamlit: Used to visualize the project on the web

This project is a web application that allows the user to input a ticker symbol and view raw stock data, such the dataframe and chart. If the ticker symbol is invalid, a message will pop up asking to input a different symbol.

Includes an option to choose the desired model type and a button to run the prediction algorithm which returns the likelihood the stock will increase tomorrow.

## How it Works
New column called "Tomorrow" is created by shifting the closing price by 1, revealing tomorrow's stock price for each day. 
By comparing the close and tomorrow columns, new column "Target" is created using 1s and 0s (1 when price goes up, 0 when price goes down)

Uses "Open", "High", "Low", "Volume", "Close" as features
Uses "Target" as the target

## Areas to improve: 
* Work on improving the accuracy of the XGBoost model by adjusting the parameters
* Complete the back testing algorithm to assess model accuracy
* Try to host the web app on a platform so it doesn't have to be hosted locally
* For future projects, add venv and vscode to gitignore file before uploading (better practice)
