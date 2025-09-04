import streamlit as st
import tensorflow as tf
import numpy as np
import json
import gdown
import os
import re
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐶",
    layout="wide",
)

MODEL_PATH = "model_mobilenetv2.h5"
CLASS_PATH = "class_indices.json"

# URL Google Drive (ganti dengan link kamu)
GDRIVE_URL = "https://drive.google.com/file/d/1t93ewP27enLqsQcyFUje4ncYpi6VxZ6A/view?usp=drive_link"

# -----------------------------
# FUNGSI AMBIL FILE ID dari LINK
# -----------------------------
def extract_drive_id(url: str) -> str:
    pattern = r"/d/([a-zA-Z0-9_-]+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        st.error("❌ Tidak bisa menemukan ID Google Drive dari link.")
        st.stop()

FILE_ID = extract_drive_id(GDRIVE_URL)
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# -----------------------------
# DOWNLOAD MODEL kalau belum ada
# -----------------------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("⬇️ Mengunduh model dari Google Drive..."):
        gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Load class indices
with open(CLASS_PATH, "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_image(image: Image.Image):
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds) * 100

    return idx_to_class[predicted_class], confidence, preds[0]

# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------
st.sidebar.title("⚙️ Pengaturan")
mode = st.sidebar.radio("Mode Prediksi", ["Single Upload", "Batch Upload"])

# -----------------------------
# MAIN TITLE
# -----------------------------
st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload gambar kucing atau anjing, lalu model akan memprediksi kelasnya.")

# -----------------------------
# SINGLE UPLOAD MODE (Drag & Drop)
# -----------------------------
if mode == "Single Upload":
    uploaded_file = st.file_uploader(
        "Tarik & lepas gambar di sini",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diupload", use_column_width=True)

        with st.spinner("🔎 Sedang memproses..."):
            pred_class, confidence, preds = predict_image(image)

        st.success(f"**Prediksi:** {pred_class}")
        st.progress(int(confidence))
        st.write(f"Confidence: **{confidence:.2f}%**")

        st.write("📊 Probabilitas tiap kelas:")
        prob_dict = {idx_to_class[i]: float(preds[i]) for i in range(len(preds))}
        st.json(prob_dict)
    else:
        st.info("Silakan upload gambar terlebih dahulu.")

# -----------------------------
# BATCH UPLOAD MODE (Multi Drag & Drop)
# -----------------------------
else:
    uploaded_files = st.file_uploader(
        "Tarik & lepas beberapa gambar sekaligus",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        cols = st.columns(2)
        for i, file in enumerate(uploaded_files):
            image = Image.open(file).convert("RGB")
            pred_class, confidence, preds = predict_image(image)

            with cols[i % 2]:
                st.image(image, caption=f"{file.name}", use_column_width=True)
                st.write(f"➡️ **{pred_class}** ({confidence:.2f}%)")
                st.progress(int(confidence))
    else:
        st.info("Silakan upload beberapa gambar untuk batch prediksi.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    """
    ✨ Dibuat dengan ❤️ menggunakan Streamlit & TensorFlow  
    Mode: **Single/Batch Upload** | Drag & Drop Support
    """
)
