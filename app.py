from numpy.core.defchararray import mod
import streamlit as st
import numpy as np
import pickle
from PIL import Image
from skimage.io import imread
from skimage.transform import resize

st.set_option('deprecation.showfileUploaderEncoding',False)

st.title("image Classification using MachineLearning")
st.text('Upload Image')

CATEGORIES = ['fresh_apple', 'fresh_banana', 'fresh_custardapple', 'fresh_grapes', 'fresh_guava', 'fresh_litchi', 'fresh_mango', 'fresh_orange', 'fresh_papaya','fresh_pneapple', 'fresh_pomegranate', 'fresh_strawberry']


model = pickle.load(open('mlDimg.pkl', 'rb'))
uploaded_file = st.file_uploader("Choose an Image: ", type=['jpg','png','jpeg','jfif'])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='uploaded image')
    
    if st.button('PREDICT'):
        st.write('Result')
        flat_data = []
        img = np.array(img)
        img_resize = resize(img,(150,150,3))
        flat_data.append(img_resize.flatten())
        flat_data = np.array(flat_data)
        y_out = model.predict(flat_data)
        y_out = CATEGORIES[y_out[0]]
        st.title(f"predicted output: {y_out}")
        
        prob = model.predict_proba(flat_data)
        for index, item in enumerate(CATEGORIES):
            st.write(f"{item}: {prob[0][index]*100} %")
