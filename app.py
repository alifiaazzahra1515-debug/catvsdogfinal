import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import gdown

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="🐱🐶 Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered",
)

# Langsung paste link share dari Google Drive (format "Anyone with the link")
GDRIVE_URL = "https://drive.google.com/file/d/1xWeBGY2lIAmxG3mqdc7Gn7sjULTAeGK8/view?usp=sharing"
MODEL_PATH = "my_tl_model.h5"
CLASS_PATH = "class_indices.json"

# -----------------------------
# DOWNLOAD MODEL
# -----------------------------
def download_from_gdrive(url, output):
    if not os.path.exists(output):
        with st.spinner("⬇️ Downloading model from Google Drive..."):
            gdown.download(url, output, quiet=False)

download_from_gdrive(GDRIVE_URL, MODEL_PATH)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_cnn_model():
    try:
        model = load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        st.stop()
    return model

model = load_cnn_model()

# Ambil ukuran input otomatis
try:
    input_height, input_width = model.input_shape[1:3]
except:
    st.error("Tidak bisa membaca input_shape dari model.")
    st.stop()

# -----------------------------
# LOAD CLASS INDICES
# -----------------------------
if os.path.exists(CLASS_PATH):
    with open(CLASS_PATH, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
else:
    st.warning("⚠️ class_indices.json tidak ditemukan, kelas tampil sebagai nomor.")
    idx_to_class = None

emoji_map = {"Cat": "🐱", "Dog": "🐶"}

# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_image(image: Image.Image):
    img_resized = image.resize((input_width, input_height))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
    preds = model.predict(img_array, verbose=0)
    probs = preds[0]

    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs) * 100)

    if idx_to_class:
        label_name = idx_to_class.get(pred_idx, str(pred_idx))
    else:
        label_name = str(pred_idx)

    label = f"{emoji_map.get(label_name, '')} {label_name}"
    return label, confidence, probs

# -----------------------------
# MAIN APP
# -----------------------------
st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload gambar kucing atau anjing untuk diprediksi.")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Gambar terupload", use_column_width=True)

    with st.spinner("🔎 Sedang memproses..."):
        label, conf, probs = predict_image(img)

    st.success(f"Prediksi: **{label}** ({conf:.2f}%)")

    # Bar chart probabilitas
    labels = [idx_to_class.get(i, str(i)) for i in range(len(probs))]
    emojis = [emoji_map.get(lbl, "") for lbl in labels]
    fig = go.Figure([go.Bar(
        x=[f"{e} {lbl}" for e, lbl in zip(emojis, labels)],
        y=probs,
        marker_color=["#1f77b4", "#ff7f0e"]
    )])
    fig.update_layout(
        title_text="Probabilitas Tiap Kelas",
        xaxis_title="Kelas",
        yaxis_title="Probabilitas",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Silakan upload gambar terlebih dahulu.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("✨ Dibuat dengan ❤️ menggunakan Streamlit & TensorFlow • Model di-load langsung dari Google Drive 🐱🐶")
