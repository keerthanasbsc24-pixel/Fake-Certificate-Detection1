
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model from Google Drive
model = tf.keras.models.load_model(
    "/content/drive/MyDrive/certificate_cnn_model.keras"
)

st.set_page_config(
    page_title="Certificate Detection",
    page_icon="📄"
)

st.title("AI-Based Detection of Fake Academic Certificates")

st.write("Upload a certificate image to get the model's prediction.")

uploaded_file = st.file_uploader(
    "Choose a certificate image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Certificate")

    if st.button("Predict"):

        img = image.resize((128, 128))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        classes = ["Fake", "Genuine"]
        result = classes[np.argmax(prediction[0])]

        st.subheader("Prediction Result")
        st.write("Predicted Class:", result)

        st.caption(
            "Classroom demonstration using synthetic sample images. "
            "This model cannot verify real academic certificates."
        )
