#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 00:49:22 2021

@author: rakshitbatra
"""
import imageio
import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps
from newspaper import Article

try:
    from googlesearch import search
except ImportError:
    print("No module named 'google' found")

st.set_page_config(page_title="CovApp", page_icon="/Users/rakshitbatra/Desktop/Covid-19_Detection/corona_favicon.png")
st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

def load_model():
    json_file = open('/Users/rakshitbatra/Desktop/Covid-19_Detection/Covid_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights("/Users/rakshitbatra/Desktop/Covid-19_Detection/Covid_model.h5")
    print("Loaded model from disk")
    loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    print("Model Compiled Succesfuly 👍")
    return loaded_model

    
model = load_model()


st.write("""
         # Covid-19 Detection Using Deep Learning
         """
         )

file = st.file_uploader("Please Upload Chest X-Rays here", type=["jpg","jpeg"])

hide_streamlit_style = """
            <style>
            footer {
	
	visibility: hidden;
	
	}
footer:after {

	content:'Rakshit Batra © 2021'; 
	visibility: visible;
	display: block;
	position: relative;
	#background-color: red;
	padding: 5px;
	top: 2px;
}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def import_and_predict(img, model):

    data = np.ndarray(shape=(1, 64, 64, 3), dtype=np.float32)
    pre_img = img
    size = (64, 64)
    pre_img = ImageOps.fit(pre_img, size=size, method=Image.ANTIALIAS)
    img_arr = image.img_to_array(pre_img)
    img_arr = np.expand_dims(img_arr, axis = 0)
    data[0] = img_arr
    rslt = model.predict(data)
    
    if rslt[0][0] == 1:
        prediction = "Normal"
    else:
        prediction = "COVID Positive"
        
    return prediction

if file is None:
    st.text("Please upload an Image file")
else:
    img = Image.open(file)
    st.image(img, use_column_width=True)
    img = img.resize((64,64))
    result = import_and_predict(img, model)
    string="Patient is "+result
    if string=="COVID Positive":
        st.error(string)
    else:
        st.success(string)
        
query = st.text_input("Enter your Query here")
      
url_link = ""
 
for j in search(query, tld="co.in", num=10, stop=1, pause=2):
    url_link += j
article = Article(url_link)

article.download()

    
article.parse()
print(article.text[:300],"...")
print("for more follow this link: ", url_link)
st.write(article.text[:300],"...")
st.write("for more follow this link: ", url_link) 

  
    
    
    