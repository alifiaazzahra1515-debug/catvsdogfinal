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
    page_title="🐱🐶 Cat vs Dog Classifier",
    page_icon="🐾",
    layout="wide",
)

MODEL_PATH = "best_cnn_model.h5"
CLASS_PATH = "class_indices.json"  # file mapping kelas (misal: {"Cat":0,"Dog":1})

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
except Exception:
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
    st.warning("⚠️ Tidak menemukan class_indices.json, kelas akan ditampilkan sebagai indeks.")
    idx_to_class = None

# -----------------------------
# EMOTICON MAP
# -----------------------------
emoji_map = {
    "Cat": "🐱",
    "Dog": "🐶"
}

# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_image(image: Image.Image):
    img_resized = image.resize((input_width, input_height))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)

    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds) * 100

    if idx_to_class:
        label = idx_to_class[predicted_class]
        label_with_emoji = f"{emoji_map.get(label, '')} {label}"
    else:
        label = str(predicted_class)
        label_with_emoji = label

    return label_with_emoji, confidence, preds[0]

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Pengaturan")
mode = st.sidebar.radio("Mode Prediksi", ["Single Upload", "Batch Upload"])

st.sidebar.markdown("---")
st.sidebar.write("📐 Input shape model:", model.input_shape)

# -----------------------------
# MAIN UI
# -----------------------------
st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload gambar kucing atau anjing, lalu model CNN akan memprediksi kelasnya.")

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
                class_label = idx_to_class[i]
                prob_dict[f"{emoji_map.get(class_label, '')} {class_label}"] = float(preds[i])
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
    Mode: **Single/Batch Upload** | Antarmuka modern dengan emoticon 🐶🐱
    """
)
