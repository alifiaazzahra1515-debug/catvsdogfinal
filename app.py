import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="🐱🐶 Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered",
)

# -----------------------------
# LOAD MODEL & CLASS INDICES
# -----------------------------
@st.cache_resource
def load_cnn_model():
    try:
        model_path = hf_hub_download(
            repo_id="alifia1/catdog1",   # ganti dengan repo kamu
            filename="mobilenetv2_single_input.h5"
        )
        model = load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        st.stop()
    return model

model = load_cnn_model()

try:
    input_height, input_width = model.input_shape[1:3]
except:
    st.error("Tidak bisa membaca input_shape dari model.")
    st.stop()

@st.cache_resource
def load_classes():
    try:
        class_path = hf_hub_download(
            repo_id="alifia1/catdog1",
            filename="class_indices.json"
        )
        with open(class_path, "r") as f:
            class_indices = json.load(f)
        return {v: k for k, v in class_indices.items()}
    except:
        st.warning("⚠️ class_indices.json tidak ditemukan, kelas tampil sebagai nomor.")
        return None

idx_to_class = load_classes()
emoji_map = {"Cat": "🐱", "Dog": "🐶"}

# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_image(image: Image.Image):
    img_resized = image.resize((input_width, input_height))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
    preds = model.predict(img_array, verbose=0)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    pred_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds) * 100)
    label = idx_to_class.get(pred_idx, str(pred_idx)) if idx_to_class else str(pred_idx)
    label = f"{emoji_map.get(label, '')} {label}"
    return label, confidence, preds[0]

# -----------------------------
# UI
# -----------------------------
st.title("🐱🐶 Cat vs Dog Classifier")
st.markdown("Upload gambar kucing atau anjing → model MobileNetV2 akan memprediksi!")

uploaded = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Gambar terupload", use_column_width=True)

    with st.spinner("🔎 Sedang memproses..."):
        label, conf, probs = predict_image(img)

    st.success(f"Prediksi: **{label}** ({conf:.2f}%)")

    # Probabilitas chart
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

st.markdown("---")
st.markdown("✨ Dibuat dengan ❤️ menggunakan Streamlit & TensorFlow • Model dihosting di Hugging Face Hub 🐱🐶")
