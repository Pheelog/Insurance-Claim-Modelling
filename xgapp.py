import streamlit as st

import xgboost as xgb
import pandas as pd
import numpy as np




#Loading up the Regression model we created

model = xgb.XGBRegressor()
model.load_model('xgb_model.json')

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


st.title("Team15's Average Claim Amount Predictor")
#st.image("""motor-insurance-policy-online.png""")
st.header('Please fill this form below')
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