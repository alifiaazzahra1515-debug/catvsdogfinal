import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ======================================================
# Konfigurasi
# ======================================================
MODEL_URL = "https://huggingface.co/alifia1/catdog1/resolve/main/model_mobilenetv2.h5"
MODEL_PATH = "model_mobilenetv2.h5"

# Download model dari Hugging Face jika belum ada
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = load_model(MODEL_PATH)

# ======================================================
# Streamlit UI
# ======================================================
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")
st.title("🐱 vs 🐶 Cat & Dog Image Classifier")
st.write("Upload gambar kucing atau anjing, dan model akan memprediksi kelasnya.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image
    img_resized = image.resize((224,224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # sesuai MobileNetV2

    # Predict
    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]
    classes = ['Cat', 'Dog']
    confidence = preds[0][class_idx]*100

    st.success(f"Predicted: **{classes[class_idx]}** with confidence {confidence:.2f}%")
