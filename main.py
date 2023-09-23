import streamlit as st
import time
from model import predict, get_data, plot_data, is_valid_ticker

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
    if st.button("Make Prediction"):
        with st.spinner("Running model..."):
            time.sleep(2)
            prediction = predict(data_set)

            #Display raw table as a table
            st.dataframe(data_set.loc[:,["Tomorrow", "Target"]], use_container_width=True)

            st.write("According to the model, there is a " + str(prediction) + "% chance that " + ticker + " will increase tomorrow" )

            if prediction > 50:
                st.markdown("Chances are the stock price will **:blue[increase]** tomorrow")
            elif prediction < 50:
                st.markdown("Chances are the stock price will **:red[decrease]** tomorrow")
elif ticker and not(is_valid_ticker(ticker)):
    st.error("Invalid or Delisted Ticker Symbol")
