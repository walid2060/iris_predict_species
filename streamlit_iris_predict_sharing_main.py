import streamlit as st
import numpy as  np
import pandas as pd
import pickle
from keras.models import load_model

loaded_model = pickle.load(open(r'C:\Users\HP\Desktop\iris_model_randomForest.sav','rb'))

st.title("Prediction Species of iris flowers")
data_new={}

min_seplen=4.3
max_seplen=7.9
moy_seplen=5.84


data_new['sepal_length']=st.slider('Enter sepal_length(in cm)',float(min_seplen),float(max_seplen),float(moy_seplen))

min_sepwid=2
max_sepwid=4.4
moy_sepwid=3.05
data_new['sepal_width'] =st.slider('Enter sepal_width(in cm)',float(min_sepwid),float(max_sepwid),float(moy_sepwid))

min_petlen=1
max_petlen=6.9
moy_pepten=3.75
data_new['petal_length']=st.slider('Enter petal_length(in cm)',float(min_petlen),float(max_petlen),float(moy_pepten))

min_petwid=0.1
max_petwid=2.5
moy_petwid=1.19
data_new['petal_width']=st.slider('Enter petal_width(in cm)',float(min_petwid),float(max_petwid),float(moy_petwid))

data_new_df = pd.DataFrame(data_new, index=[0])

# Make prediction
if st.button('Predict species iris'):
        
        #data_new_scal=std.fit_transform(data_new_df)

        pred = loaded_model.predict(data_new_df)
        if pred[0] == 0:
            st.success('The predicted type of iris flower is  Setosa')
        elif pred[0] ==1:
        

            st.success('The predicted type of iris flower is  Versicolor')
        else:
            st.success('The predicted type of iris flower is  Virginica ')
            

# Run the web app