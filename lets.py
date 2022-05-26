import numpy as np
import pickle
import streamlit as st

loader_model = pickle.load(open('trained_model.sav','rb'))
## creating a fucntion for prediction
def heart_disease(input_data):
    #change input data in numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array as we are predicting for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loader_model.predict(input_data_reshaped)
    print(prediction)
    # type(prediction)

    if(prediction[0] == 0):
        return 'The person has no heart disease'
    else:
        return 'The person has disease'


def main():

    # title
    st.title('Disease prediction web app')

    # getting input data form user
    age = st.text_input('Enter age: ')
    sex = st.text_input('Enter sex: ')
    cp = st.text_input('Enter cp: ')
    trestbps = st.text_input('Enter trestbps: ')
    chol = st.text_input('Enter chol: ')
    fbs = st.text_input('Enter fbs: ')
    restecg = st.text_input('Enter restecg: ')
    thalach = st.text_input('Enter thalach: ')
    exang = st.text_input('Enter exang: ')
    oldpeak = st.text_input('Enter oldpeak: ')
    slope = st.text_input('Enter slope: ')
    ca = st.text_input('Enter ca: ')
    thal = st.text_input('Enter thal: ')


    # input_data = (55,0,0,128,205,0,3,130,1,2,1,1,3)
    diagnosis = ' '

    ## creating a button for prediction

    if st.button('Results'):
        diagnosis = heart_disease([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])

    st.success(diagnosis)



if __name__ == '__main__':
    main()




    #
