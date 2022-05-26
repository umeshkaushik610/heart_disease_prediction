import numpy as np
import pickle

loader_model = pickle.load(open('D:\ML\projects\heart_diseases\trained_model.sav','rb'))

input_data = (55,0,0,128,205,0,3,130,1,2,1,1,3)
#change input data in numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loader_model.predict(input_data_reshaped)
print(prediction)
# type(prediction)

if(prediction[0] == 0):
    print('The person has no heart disease')
else:
    print('The person has disease')