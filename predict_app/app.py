import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model from the trained session and set up classes/categories
@st.cache
def fetch_model():
    return tf.keras.models.load_model('models/cifar_cnn1.h5')

loaded_model = fetch_model()

class_names = ['airplane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Define a function to process an image and output a category in class_names
def pred_img(x):
    img1 = Image.open(x).convert(mode="RGB")
    img1 = img1.resize((32,32))
    array1 = np.array(img1.getdata())
    img_np_array = np.reshape(array1, (32,32,3)) / 255.0
    return class_names[np.argmax(loaded_model.predict(np.expand_dims(img_np_array, axis=0)))]

# Define the user interface
def app():
    st.title('CIFAR Image Classification')
    st.write('Use a CNN to classify images into general categories')
    st.markdown('----')

    col1, col2 = st.beta_columns([2,1])

    with col1:
        raw_image = st.file_uploader('Upload an Image')

        if raw_image:
            st.image(raw_image)

    with col2:
        st.write('Make a Prediction')
        if st.button('Run Model') and raw_image:
            st.write(f'`Prediction: ` {pred_img(raw_image)}')

if __name__ =='__main__':
    app()