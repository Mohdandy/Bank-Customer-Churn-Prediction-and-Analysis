import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

# set page configuration
st.set_page_config(
    page_title= 'churn_EDA',
    layout='wide',
    initial_sidebar_state='expanded'
)


# create function for EDA
def run():

    # create title
    st.title('Bank Customer Churn EDA')

    # create sub header
    st.subheader('Exploration Data Analysis for Dataset Bank Customer Churn Prediction')

    # add image
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTsbB603TUoqcvSIhl4dMoCN3T6EZvpczGnBA&usqp=CAU', use_column_width=True,
             caption='Bank Customer Churn')

    # create a description
    st.write('by Mohammad Dandy')
    st.write('# Introduction to EDA')
    st.write('It starts by displaying a dataframe')

    # Magic Syntax
    st.write('''
    On this page, I will do Exploratory Data Analysis,
    Using the dataset Bank Customer Churn.
    This dataset comes from Kaggle
    ''')
    # create straight line
    st.markdown('---')

    # show dataframe
    df = pd.read_csv('P1M2_dandy.csv')
    st.dataframe(df)
    # Menampilkan jumlah baris dan kolom
    st.write(f"Jumlah Baris: {df.shape[0]}")
    st.write(f"Jumlah Kolom: {df.shape[1]}")
    
# create straight line
    st.markdown('---')
    st.subheader('Churn Distribution')
   # Data for the pie chart
    proc = [df['churn'].value_counts().get(0, 0), df['churn'].value_counts().get(1, 0)]

    # Create a two-column layout
    col1, col2 = st.columns(2)
    
    # Plot the pie chart in the first column
    with col1:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(proc,
               labels=['stay', 'churn'],
               autopct='%1.1f%%',
               startangle=90,
               explode=(0.1, 0),
            #    colors=['#00f900', 'red'],
               wedgeprops={'edgecolor': 'black', 'antialiased': True})
        plt.title('stay vs churn percentage')
        st.pyplot(fig)

    # Plot the countplot in the second column
    with col2:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax = sns.countplot(x=df['churn'],
                           hue=df['churn'],
                        #    palette=['#00f900', 'red'],
                           edgecolor="black")
        for rect in ax.patches:
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 2,
                    rect.get_height(),
                    horizontalalignment='center',
                    fontsize=11)
        plt.title('stay vs churn count')
        st.pyplot(fig)
    # # create straight line
    # st.markdown('---')
    
    # perc_churn = df['churn'].sum() / len(df['churn'])
    # st.write(f'Churn Percentage = 1 is {perc_churn * 100:.1f}%')

    # # Create a pie chart
    # fig, ax = plt.subplots(figsize=(25, 8))
    # df['churn'].value_counts().plot(kind='pie', autopct="%1.1f%%", ax=ax)
    # ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # st.pyplot(fig)

    # # Melihat total value
    # total_values = list(df['churn'].value_counts().values)
    # st.write(f'Total Values: {total_values}')
    st.write('''
    From the chart, it can be seen that 20.4% or 2037 people will churn or leave, while 79.6% or 7963 people will not churn or not leave.
    ''')

    st.subheader('Country Distribution')
    # Calculate country counts
    country_counts = df['country'].value_counts(normalize=True)
    labels = country_counts.index
    sizes = country_counts.values

    # Create a pie chart with specified figsize
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.pie(sizes, labels=labels, autopct='%.1f%%', startangle=90)
    ax.set_title('Countries')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Display the pie chart using Streamlit
    st.pyplot(fig)
    st.write('''
    From the Pie chart it can be seen that France dominates the data with 50.1% followed by Germany with 25.1 and Spain 24.8%.
    ''')
       
    # create straight line
    st.markdown('---')
    # Display Countries Credit Score Comparison as a subheader
    st.subheader('Countries Credit Score Comparison')

    # Create a boxplot  
    fig, ax = plt.subplots(figsize=(25, 10))
    sns.boxplot(x='country', y='credit_score', data=df, ax=ax)
    ax.set_title('Countries Credit Score Comparison')

    # Display the boxplot using Streamlit
    st.pyplot(fig)
    # # Countries Credit Score Comparison
    # st.subheader('Countries Credit Score Comparison')
    # sns.boxplot(x='country', y='credit_score', data=df)
    # plt.title('Countries Credit Score Comparison')
    # st.pyplot(fig)
    st.write('''
    From the boxplot, it seems there is no significant difference for `credit score` data. But for the outliers there seems more outliers in Spain than others 2.
    Also for Germany there is the lowest value below 400 for `credit score` that is not outliers.''')
    # Age Distribution of Churned Customers
    # Display Age Distribution of Churned Customers as a subheader
    st.subheader('Age Distribution of Churned Customers')

    # Create a histogram using Seaborn
    fig, ax = plt.subplots(figsize=(25, 10))
    sns.histplot(data=df, x='age', hue='churn', multiple='stack', bins=10, ax=ax)
    ax.set_title('Age Distribution of Churned Customers')

    # Display the histogram using Streamlit
    st.pyplot(fig)
    # st.subheader('Age Distribution of Churned Customers')
    # fig, ax = sns.histplot(data=df, x='age', hue='churn', multiple='stack', bins=10, palette='viridis')
    # plt.title('Age Distribution of Churned Customers')
    # st.pyplot(fig)
    st.write('''
    From the barchart, `age` of churned customer gathered between 20-70. The biggest churned customer is 40-50. 
    Also for loyal customer, age 30-40 is the biggest loyal customer. Also the `age` distribution is right skewed.
    ''')
# create straight line
    st.markdown('---')
    st.subheader('Correlation Heatmap Feature vs churn')
   # Create a correlation heatmap
    # Define columns to exclude
    columns_to_exclude = ['country','gender']  # Add the column names you want to exclude
    df_try =df.drop(columns= columns_to_exclude, errors='ignore')
    plt.figure(figsize=(15, 5))
    sns.heatmap(df_try.corr(),
                cmap='coolwarm',
                annot=True)
    plt.xticks(rotation=50, ha='right')
    plt.title('Correlation Heatmap')
   
    # Display the heatmap using Streamlit
    st.pyplot(plt)
    st.write(''' 
    There is no significant correlation between variable to target.
    ''')
    ''' 
    There is no significant correlation between variable to target.
    '''

if __name__ == '__main__':
    run()
    
