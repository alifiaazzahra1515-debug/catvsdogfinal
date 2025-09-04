import os
import json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests

MODEL_URL = "https://huggingface.co/alifia1/catvsdog/resolve/main/model_mobilenetv2.keras"
CLASS_URL = "https://huggingface.co/alifia1/catvsdog/resolve/main/class_indices.json"
MODEL_PATH = "model_mobilenetv2.keras"
CLASS_PATH = "class_indices.json"

# Download file jika belum ada
def download_file(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename} ..."):
            r = requests.get(url)
            with open(filename, "wb") as f:
                f.write(r.content)

download_file(MODEL_URL, MODEL_PATH)
download_file(CLASS_URL, CLASS_PATH)

# Load model (cache supaya tidak reload tiap kali)
@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

model = load_my_model()

# Load class indices
with open(CLASS_PATH, "r") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

def prepare_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Streamlit UI
st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload gambar kucing atau anjing untuk diprediksi.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = prepare_image(img)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]
    label = idx_to_class[class_idx]
    confidence = float(np.max(preds))

    st.subheader("Prediction Result")
    st.success(f"**Class:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}")
