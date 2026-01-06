
import numpy as np
import streamlit as st
import pickle

#load model

def load_model():
    try:
        with open('svm.pkl','rb') as file:
            model=pickle.load(file)

        scaler=None
        try:
            with open('scaler.pkl','rb') as file:
                scaler=pickle.load(file)
        except:
            st.warning('Scaler not found')

        return model,scaler
    except FileNotFoundError as e:
        return f'Model file is not found: {e}'

model,scaler=load_model()

#convert gender to numerical

def genderinput(gender_input):
    if gender_input=='Male':
        return 0
    else:
        return 1


#convert predictions to text

def resultout(result):
    if result==1:
        return 'Yes'
    else:
        return 'No'


#function for prediction

def customer_predict(gender_input,age_input,salary_input):
    try:
        gender_value=genderinput(gender_input)
        age_value=float(age_input)
        salary_value=float(salary_input)

        input_data=np.array([[gender_value,age_value,salary_value]])

        if scaler is None or not hasattr(scaler,'transform'):
            return 'Error scaler not available or invalid'

        scaled_data=scaler.transform(input_data)

        prediction=model.predict(scaled_data)
        probabilities=model.predict_proba(scaled_data)
        predicted_purchase=int(prediction[0])

        confidence=probabilities[0][predicted_purchase]

        return predicted_purchase, confidence
    except Exception as e:
        return f"Prediction Error: {e}", None


#main

st.title('Customer Behaviour Predction App')
gender_input=st.selectbox('Select Gender',['Male','Female'])
age_input=st.number_input('Enter Age',min_value=18,max_value=100,value=30) 
salary_input=st.number_input('Enter Estimated Salary',min_value=0,max_value=5000,step=500) 

if st.button('Predict Customer Purchase'):
    if model is None:
        st.error('Model not loaded properly. Please check the file')

    result, confidence=customer_predict(gender_input,age_input,salary_input)

    if isinstance(result,str) and (result.startswith('Error') or result.startswith('Prediction Error')):
        st.error(result)
    else:
        result_output=resultout(result)
        if result_output=='Yes':
            st.success(f'Will Customer Purchase ? : {result_output}')
        else:
            st.error(f'Will Customer Purchase ? : {result_output}') 

        if confidence is not None:
            st.info(f'Confidence : {confidence:.2%}')
