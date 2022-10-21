#loading necessary libraries
import streamlit as st
import pickle
import xgboost as xgb
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
import numpy as np

#Loading up reggression model

#linear model


#xgboost model
model = xgb.XGBRegressor()
model.load_model('xgb_model.json')

#setting up side bar
with st.sidebar:
    pic = Image.open('pic4.png')
    st.image(pic)

    selected = option_menu('Insurance Claim Prediction System',
    
                        ['Home','Metrices and Graphs',
                        'Linear Model Prediction',
                        'XGBoost Model Prediction'],
                        icons=['heart','activity','activity','activity'],
                        default_index=0)
    

    st.write('')
    st.markdown('---')
    st.write('**| Team 18:**')
    st.write('**This WebApp was developed by Explore AI Team 18 (Interns).**')

#Home
if(selected == 'Home'):
    st.title('Insuance Claim Prediction System')
    data = pd.read_csv('cleandata.csv')

    col1, col2= st.columns(2)

    with col1:
        pic = Image.open('pic2.png')
        st.image(pic)
    
    with col2:
        """
        ##  | The Insurance Claim Predictor
        The Insurance Claim Predictor is a Web App developed to help insurance companies predict
        the claim amount of prospective customers, just fill in the form and hit the *Submit* button.
        ##  | Our data
        The data used for training were gotten from Explore AI.
        260894 records were collected for model training.
        """

    # Create columns for the data summary
    # column 1
    with col2:
        st.title(round(48191.09))
        st.text('AVERAGE CLAIM AMOUNT')

    with col2:
        st.title(292990)
        st.text('NUMBER OF POLICY HOLDERS')
        

# Metrices and Graphs Page
if(selected == 'Metrices and Graphs'):

    # page title
    st.title('Metrices and Graphs')

    st.header('Insurance Claim DataSet')
    data = pd.read_csv('cleandata.csv')
    st.write(data.head())
    st.caption('Dataset used for building Insurace Claim Model')

    st.header('Model Performance Evaluation')

    df = pd.DataFrame({
            "Model/Metrics": ["Linear Regression", 'XGBoost'],
            "MAE":[7416.35, 7774.62],
            "RMSE":[24271.48, 23551.36],
            "Explained Variance":[0.044, 0.102]
        })
    st.table(df)
    st.caption('Result of moddel performance evaluation')

    st.subheader('Actual Vs Linear Regresssion Predicted Claim Amount')

    image = Image.open('linear.png')
    st.image(image, caption='Actual vs Linear Regression Predicted Claims Amount Distribution')

    st.subheader('Actual Vs XGBoost Regresssion Predicted Claim Amount')

    xgbimage = Image.open('xgb.png')
    st.image(xgbimage, caption='Actual vs XGBoost Predicted Claims Amount Distribution')


if(selected == 'Linear Model Prediction'):

    # page title
    st.title('Linear Model Prediction')

# XGBoost Model Prediction Page
if(selected == 'XGBoost Model Prediction'):

    # page title
    st.title('XGBoost Model Prediction')

    #Caching the model for faster loading
    @st.cache


# Define the prediction function
    def predict(Gender, Age, VehicleAge, Model, Colour, CubicCapacity, Kilowats):
    #Predicting the insurance amount
        if Model == 'volkswagen polo':
            Model = 0
        elif Model == 'hyundai grand':
            Model = 1
        elif Model == 'dacia sandero':
            Model = 2
        elif Model == 'nissan np200':
            Model = 3
        elif Model == 'toyota etios':
            Model = 4
        elif Model == 'hyundai i20':
            Model = 5
        elif Model == 'ford fiesta':
            Model = 6
        elif Model == 'toyota corolla':
            Model = 7
        elif Model == 'renault kwid':
            Model = 8
        elif Model == 'kia picanto':
            Model = 9
        elif Model == 'other':
            Model = 10
    
        if Colour == 'grey':
            Colour = 0
        elif Colour == 'silver':
            Colour = 1
        elif Colour == 'white':
            Colour = 2
        elif Colour == 'blue':
            Colour = 3
        elif Colour == 'gold':
            Colour = 4
        elif Colour == 'charcoal':
            Colour = 5
        elif Colour == 'black':
            Colour = 6
        elif Colour == 'red':
            Colour = 7
        elif Colour == 'brown':
            Colour = 8
        elif Colour == 'other':
            Colour = 9
    
    
        if Gender == 'male':
            Gender = 0
        elif Gender == 'female':
            Gender = 1

    

        prediction = model.predict(pd.DataFrame([[Gender, Age, VehicleAge, Model, 
                                              Colour, CubicCapacity, Kilowats]], 
                                            columns=['Gender', 'Age', 'VehicleAge', 
                                                     'Model', 'Colour', 
                                                     'CubicCapacity', 'Kilowatts']))
        return prediction


    st.subheader('Fill the form to predict the claim amount of a policy holder')
    Age = st.slider('What is your age?', 18, 100)
    Model = st.selectbox('Vehicle Model', ['volkswagen polo', 'other', 'hyundai grand', 'dacia sandero', 
                                      'nissan np200', 'toyota etios', 'hyundai i20', 'ford fiesta', 
                                      'toyota corolla', 'renault kwid', 'kia picanto'])
    Colour = st.selectbox('Vehicle Colour:', ['grey', 'silver', 'white', 'blue', 'other', 'gold', 
                                         'charcoal', 'black', 'red', 'brown'])
    Gender = st.selectbox('Gender', ['male', 'female'])
    VehicleAge = st.slider('What is the age of your vehice?', 1, 60)
    CubicCapacity = st.number_input('Cubic Capacity', min_value=100.00, max_value=100000.00, value=100.00)
    Kilowats = st.number_input('Kilowatts', min_value=1.00, max_value=10000.00, value=1.00)

    if st.button('Predict Amount'):
        price = predict(Gender, Age, VehicleAge, Model, Colour, CubicCapacity, Kilowats)
        st.success(f'The predicted average claim amount is R{price[0]:.2f} ')