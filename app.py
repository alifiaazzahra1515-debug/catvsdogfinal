import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="🐱🐶 Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered",
)

MODEL_PATH = "best_cnn_model.h5"
CLASS_PATH = "class_indices.json"

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
# INFO SECTIONS
# -----------------------------
st.markdown("## 🟦 1. About")
st.write("""
Hi all, welcome to this project 👋  
This is a **Cat or Dog Recognizer App** built using **Convolutional Neural Networks (CNN)** and **Streamlit**. 🐱🐶

👉 Tujuan aplikasi ini adalah untuk mengklasifikasi gambar **kucing** atau **anjing** secara otomatis.  
Dengan antarmuka sederhana, siapapun bisa menggunakannya tanpa perlu paham machine learning.
""")

st.markdown("## 🟦 2. How To Use It")
st.write("""
Menggunakan aplikasi ini sangat mudah:  

1. 📥 **Upload gambar** kucing 🐱 atau anjing 🐶 (format JPG/PNG).  
2. 🖱️ Bisa klik *Browse files* atau **drag & drop** ke kotak upload.  
3. ✅ Pastikan file benar-benar **gambar**, bukan dokumen lain.  
4. 🔎 Tunggu sebentar → model akan memproses dan menampilkan hasil.  
5. 📊 Hasil prediksi dilengkapi dengan **confidence score** dan **visualisasi probabilitas**.  

**NOTE:** Kalau upload file bukan gambar, aplikasi akan menampilkan pesan error 🚫.
""")

st.markdown("## 🟦 3. What It Will Predict")
st.write("""
Model ini akan memprediksi apakah gambar yang kamu upload adalah:  

- 🐱 **Cat**  
- 🐶 **Dog**

Selain label prediksi, aplikasi juga menampilkan:  
- Persentase **confidence** (keyakinan model).  
- Grafik probabilitas tiap kelas untuk transparansi hasil prediksi.  
""")

# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_image(image: Image.Image):
    img_resized = image.resize((input_width, input_height))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
    preds = model.predict(img_array, verbose=0)

    # Ambil probabilitas (1D array)
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
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Pengaturan")
mode = st.sidebar.radio("Mode Prediksi", ["Single Upload", "Batch Upload"])
st.sidebar.markdown("---")
st.sidebar.write("📐 Input shape model:", model.input_shape)

# -----------------------------
# MAIN APP
# -----------------------------
st.markdown("## 🟦 4. Try It Out")

if mode == "Single Upload":
    uploaded = st.file_uploader("Tarik & lepas gambar di sini", type=["jpg", "jpeg", "png"])

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

else:  # Batch Upload
    uploaded_files = st.file_uploader(
        "Upload beberapa gambar sekaligus",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        cols = st.columns(2)
        for i, file in enumerate(uploaded_files):
            img = Image.open(file).convert("RGB")
            label, conf, _ = predict_image(img)
            with cols[i % 2]:
                st.image(img, caption=f"{file.name}", use_column_width=True)
                st.write(f"➡️ **{label}** ({conf:.2f}%)")
                st.progress(int(conf))
    else:
        st.info("Silakan upload beberapa gambar untuk batch prediksi.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("✨ Dibuat dengan ❤️ menggunakan Streamlit & TensorFlow • UI modern, informatif & interaktif 🐱🐶")
