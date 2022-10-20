#Import all libraries
from unittest import result
from sklearn import datasets
import streamlit as st
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import pickle
import joblib

#Create Streamlit app title
st.title('Linear Model Prediction')

#Create input widgets
col1, col2 = st.columns(2)

with col1:
    PreviousInsurerExcess = st.number_input('Previous Excess')
        
with col2:
    PreviousInsurerPremium = st.number_input('Previous Premium')
            
with col1:
    Occupation = st.selectbox('Occupation', ['Others', 'Educator', 'Driver', 'Nurse', 'Self_Employed','Police_Officer', 'Supervisor', 'Manager', 'Teacher', 'Operator'])

with col2:
    IndustryType = st.selectbox('Industry', ['Office_Clerical_Manager', 'Government', 'Others', 'Professional','Government_Education', 'Medical', 'Mining', 'Construction','Charity'])

with col1:
    Gender = st.selectbox('Gender', ['Female', 'Male'])
        
with col2:
    MaritalStatus = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Widowed'])

with col1:
    Make = st.selectbox('Vehicle Make', ['Volkswagen', 'Kia', 'Bmw', 'Hyundai', 'Others', 'Renault','Toyota', 'Nissan', 'Ford', 'Chevrolet'])
        
with col2:
    Model = st.selectbox('Vehicle Model', ['Polo', 'Others', 'Grand', 'Sandero', 'Etios', 'I20', 'Fiesta','Corolla', 'Kwid', 'Picant'])

with col1:
    Colour = st.selectbox('Vehicle Colour', ['Grey', 'Silver', 'White', 'Blue', 'Others', 'Gold', 'Black','Red', 'Brown'])        

with col2:
    Transmission = st.selectbox('Vehicle Transmission',['M', 'A'])
        
with col1:
    VehicleType = st.selectbox(' Vehicle Type', ['Auto', 'Light_Commercial'])
        
with col2:
    BodyType = st.selectbox('Vehicle Body Type', ['H_B', 'S_D', 'X_O', 'D_C', 'S_C', 'SUV', 'Others', 'MPV', 'C_P','C_B'])
        
with col1:
    Kilowatts = st.number_input('Vehicle Kilowatts')

with col2:
    VehicleYear = st.number_input('Vehicle Year')
    
with col1:
    PersonProvince = st.selectbox('Province', ['Kwazulu_Natal', 'Gauteng', 'Mpumalanga', 'Eastern_Cape','Limpopo', 'Free_State', 'North_West', 'Western_Cape','Northern_Cape'])
    
with col2:
    SumAssured = st.number_input('Vehicle Retail Value')
    
with col1:
    Age = st.number_input('Age')

dataset = pd.read_csv('modeldata.csv')
dataset_categorical = dataset.drop(['ClaimAmount','PreviousInsurerExcess','PreviousInsurerPremium', 'Kilowatts', 'VehicleYear', 'Age'], axis=1)
dataset_numerical = dataset.drop(['Occupation', 'IndustryType', 'Gender', 'MaritalStatus','Make','Model','Colour','Transmission','VehicleType','BodyType','PersonProvince'], axis=1)

    #Create a dictionary that maps user input
input_dict = {'PreviousInsurerExcess':PreviousInsurerExcess,'PreviousInsurerPremium':PreviousInsurerPremium, 'Occupqtion':Occupation, 'IndustryType':IndustryType, 'Gender':Gender, 
'MaritalStatus':MaritalStatus,'Make':Make,'Model':Model,'Colour':Colour,'Transmission':Transmission,'VehicleType':VehicleType,'BodyType':BodyType,'Kilowatts':Kilowatts, 'VehicleYear':VehicleYear, 
'PersonProvince':PersonProvince,'Age':Age}
input_df = pd.DataFrame([input_dict])


#derive the dataframe of categorical feature alone
input_df_cat = input_df.drop(['PreviousInsurerExcess','PreviousInsurerPremium', 'Kilowatts', 'VehicleYear', 'Age'], axis = 1)

#expansion of columns using one hot encoding
expanded_columns = ['Occupation_Educator','Occupation_Manager', 'Occupation_Nurse', 'Occupation_Operator','Occupation_Others', 'Occupation_Police_Officer','Occupation_Self_Employed', 'Occupation_Supervisor',
       'Occupation_Teacher', 'IndustryType_Construction','IndustryType_Government', 'IndustryType_Government_Education','IndustryType_Medical', 'IndustryType_Mining','IndustryType_Office_Clerical_Manager', 
       'IndustryType_Others','IndustryType_Professional', 'Gender_Male', 'MaritalStatus_Married','MaritalStatus_Single', 'MaritalStatus_Widowed', 'Make_Chevrolet','Make_Ford', 'Make_Hyundai', 'Make_Kia', 
       'Make_Nissan', 'Make_Others','Make_Renault', 'Make_Toyota', 'Make_Volkswagen', 'Model_Etios','Model_Fiesta', 'Model_Grand', 'Model_I20', 'Model_Kwid','Model_Others', 'Model_Picanto', 'Model_Polo', 
       'Model_Sandero','Colour_Blue', 'Colour_Brown', 'Colour_Gold', 'Colour_Grey','Colour_Others', 'Colour_Red', 'Colour_Silver', 'Colour_White','Transmission_M', 'VehicleType_Light_Commercial', 
       'BodyType_C_P','BodyType_D_C', 'BodyType_H_B', 'BodyType_MPV', 'BodyType_Others','BodyType_SUV', 'BodyType_S_C', 'BodyType_S_D', 'BodyType_X_O','PersonProvince_Free_State', 
       'PersonProvince_Gauteng','PersonProvince_Kwazulu_Natal', 'PersonProvince_Limpopo','PersonProvince_Mpumalanga', 'PersonProvince_North_West','PersonProvince_Northern_Cape', 'PersonProvince_Western_Cape']

new_df = pd.get_dummies(input_df_cat).reindex(columns=expanded_columns, fill_value=0)

new_df1 = pd.concat([new_df,input_df],axis=1)

#import joblib
pt_model = joblib.load('lr')

def predicter():
    m = pt_model.predict(new_df1)
    return m

predict_button = st.button('Predict Claim', on_click=predicter)

if predict_button:
    result = predicter()
    st.success(f'The predicted average claim amount is ${result[0]:.2f} USD')
