import streamlit as st
import EDA_M2_dandy
import prediction_M2

navigation = st.sidebar.selectbox('Select Page: ', ('EDA','Predict Churn'))

if navigation == "EDA":
    EDA_M2_dandy.run()
else:
    prediction_M2.run()