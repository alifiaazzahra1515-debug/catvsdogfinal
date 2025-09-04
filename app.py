import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
from tensorflow.keras.models import load_model

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Image Classifier",
    page_icon="🖼️",
    layout="wide",
)

MODEL_PATH = "best_cnn_model.h5"
CLASS_PATH = "class_indices.json"  # harus ada file json mapping kelas

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

# -----------------------------
# LOAD CLASS INDICES
# -----------------------------
if os.path.exists(CLASS_PATH):
    with open(CLASS_PATH, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
else:
    st.warning("⚠️ Tidak menemukan class_indices.json, kelas akan ditampilkan sebagai indeks.")
    idx_to_class = None

# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_image(image: Image.Image):
    img_resized = image.resize((224, 224))  # ukuran default CNN kamu
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)

    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds) * 100

    if idx_to_class:
        label = idx_to_class[predicted_class]
    else:
        label = str(predicted_class)

    return label, confidence, preds[0]

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Pengaturan")
mode = st.sidebar.radio("Mode Prediksi", ["Single Upload", "Batch Upload"])

# -----------------------------
# MAIN UI
# -----------------------------
st.title("🖼️ Image Classifier with CNN")
st.write("Upload gambar, lalu model CNN akan memprediksi kelasnya.")

# -----------------------------
# SINGLE UPLOAD
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
        prob_dict = {}
        for i in range(len(preds)):
            if idx_to_class:
                prob_dict[idx_to_class[i]] = float(preds[i])
            else:
                prob_dict[str(i)] = float(preds[i])
        st.json(prob_dict)
    else:
        st.info("Silakan upload gambar terlebih dahulu.")

# -----------------------------
# BATCH UPLOAD
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
    Mode: **Single/Batch Upload** | Antarmuka modern & mudah digunakan
    """
)
