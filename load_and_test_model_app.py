import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json

def load_class_names1(filename):
    with open(filename, 'r') as file_name:
        return json.load(file_name)

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('./best_model.h5')
    return model

@st.cache
def load_class_names(filename):
    with open(filename, 'r') as file_name:
        return json.load(file_name)


def predict(image, model):
    image = image.resize((128, 128)) # assuming this is the size your model expects
    image = np.array(image) #/ 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)[0]
    predicted_class_id = int(np.argmax(prediction)) #Conver from numpy int to int to be able to use it as index
    classes_list = load_class_names("class-labels.json")
    predicted_class = classes_list[predicted_class_id]
    prediction_score = prediction[predicted_class_id]
    return predicted_class_id, predicted_class, prediction_score


model = load_model()

st.title('Image Classification')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.text(f'{type(image)}')
    predicted_class_id, predicted_class, prediction_score = predict(image, model)
    st.write(f'predicted_class_id: {predicted_class_id}')
    st.write(f'predicted_class: {predicted_class}')
    st.write(f'prediction score: {prediction_score}')
