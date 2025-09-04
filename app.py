import streamlit as st
import tensorflow as tf
import numpy as np
import json
import gdown
import os
import re
from PIL import Image
from tensorflow.keras.models import Model

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

# URL Google Drive model
GDRIVE_URL = "https://drive.google.com/file/d/1t93ewP27enLqsQcyFUje4ncYpi6VxZ6A/view?usp=drive_link"

# -----------------------------
# HELPER: extract ID dari URL
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
    try:
        base_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        # ⚡ force sub-model → hanya pakai input[0] ke output terakhir
        if isinstance(base_model.inputs, (list, tuple)) and len(base_model.inputs) > 1:
            st.warning("Model punya >1 input, hanya input pertama yang dipakai.")
            model = Model(inputs=base_model.inputs[0], outputs=base_model.outputs)
        else:
            model = base_model
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        st.stop()
    return model

model = load_model()

# -----------------------------
# LOAD CLASS INDICES
# -----------------------------
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

    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds) * 100
    return idx_to_class[predicted_class], confidence, preds[0]

# -----------------------------
# UI
# -----------------------------
st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload gambar kucing atau anjing, lalu model akan memprediksi kelasnya.")

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
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    """
    ✨ Dibuat dengan ❤️ menggunakan Streamlit & TensorFlow  
    (Versi anti-error: hanya pakai input pertama model)
    """
)
