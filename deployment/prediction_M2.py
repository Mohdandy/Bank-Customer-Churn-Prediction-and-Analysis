import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Load Model

# Load file pipeline pkl
with open('best_dt.pkl', 'rb') as file_1:
    best_dt = pickle.load(file_1)

# Load file preprocessor
with open('preprocessor.pkl', 'rb') as file_2:
    preprocessor = pickle.load(file_2)

def run():
    
    products_number = st.selectbox(label='Select your product number:', options=[1, 2, 3, 4])
    age = st.number_input(label='Select your age',min_value=18,max_value=91) # since min age is 18 and max 91
    active_member = st.selectbox(label="Select your status member", options=[0,1])
    balance = st.number_input(label='Select your balance',min_value=0.00,max_value=296710.00)
    gender = st.radio(label="Select your gender", options=["Male","Female" ]) # male, female
    country = st.radio(label="Select your country", options=["Spain", "Germany", "France"]) # "Spain", "Germany", "France"
    credit_score = st.number_input(label='Select your credit score',min_value=350,max_value=850)
    tenure = st.number_input(label='Select your tenure',min_value=0,max_value=10)
    credit_card = st.selectbox(label="Select your credit card status", options=[0,1])

    df_inf = pd.DataFrame({
        'products_number' : products_number,
        'age' : age,
        'active_member' : active_member,
        'balance' : balance,
        'gender' : gender,
        'country': country,
        'credit_score': credit_score ,
        'tenure' : tenure,
        'credit_card' : credit_card
        
        }, index =[0])

    st.table(df_inf)
    
    if st.button(label='Predict'):
    
        # Prediction
        # define df_inf_final trough preprocessor
        df_inf_final = preprocessor.transform(df_inf)
        inf_pred = best_dt.predict(df_inf_final)

        st.write(inf_pred[0])

        if inf_pred[0] == 0:
            st.write('Customer will likely leave')
        else:
            st.write('Customer will likely stay')

if __name__== '__main__':
   run()