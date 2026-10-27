import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Load model
model = tf.keras.models.load_model("vegetable_cnn.keras")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Title
st.title("Vegetable Image Classifier")

st.write("Upload a vegetable image")

# Upload image
file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if file is not None:

    image = Image.open(file).convert("RGB")

    st.image(image, caption="Uploaded Image")

    if st.button("Predict"):

        image = image.resize((128, 128))

        image = np.array(image) / 255.0

        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image)

        index = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        st.success("Prediction: " + class_names[index])

        st.write("Confidence:", round(confidence, 2), "%")