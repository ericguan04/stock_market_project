import streamlit as st
import time
from model import predictXGBoost, predictRandomForest, get_data, plot_data, is_valid_ticker, trendFeatures

st.set_page_config(page_title="Stock Market Prediction", page_icon=":chart_with_upwards_trend:", layout="wide")

st.title("Stock Market Prediction")
ticker = st.text_input("Enter ticker symbol")

#Checks to make sure a ticker symbol is entered and its validity before displaying data
#Reports and error if ticker symbol entered is not valid
if ticker and is_valid_ticker(ticker):
    st.divider()
    #Get the data set from ticker symbol
    st.write("Data Frame")
    data_set = get_data(ticker)
    
    #Display raw table as a table
    st.dataframe(data_set, use_container_width=True)
    #Display data as a chart
    fig = plot_data(data_set)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    #Choose the ML prediction model used
    model = st.selectbox("Select Prediction Model", ("XGBoost Regressor", "Random Forest Classifier"))

    if st.button("Make Prediction"):
        with st.spinner("Running model..."):
            time.sleep(1.5)
            #Create new features and update the data set
            predictors, data_set = trendFeatures(data_set)
            if model == "XGBoost Regressor":
                prediction = predictXGBoost(data_set, predictors)
            elif model == "Random Forest Classifier":
                prediction = predictRandomForest(data_set, predictors)

            #Display data used by the ML model for prediction
            st.dataframe(data_set.loc[:,["Tomorrow", "Target"] + predictors], use_container_width=True)

            st.write("According to the model, there is a " + str(prediction) + "% chance that " + ticker + " will increase tomorrow" )

            if prediction > 50:
                st.markdown("Chances are the stock price will **:blue[increase]** tomorrow")
            elif prediction < 50:
                st.markdown("Chances are the stock price will **:red[decrease]** tomorrow")
elif ticker and not(is_valid_ticker(ticker)):
    st.error("Invalid or Delisted Ticker Symbol")
