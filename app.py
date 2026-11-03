
import os
import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from PIL import Image

st.set_page_config(
    page_title="Certificate Detection",
    page_icon="📄"
)

st.title("AI-Based Detection of Fake Academic Certificates")
st.write("Upload a certificate image to see the model's prediction.")

MODEL_PATH = "certificate_cnn_model.keras"
FILE_ID = "1vscSckeSogL9whm_S4Qc6mxpfYi8e2eU"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Certificate Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Certificate",
             use_container_width=True)

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
