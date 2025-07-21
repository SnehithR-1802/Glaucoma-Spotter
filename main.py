import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

st.set_page_config(page_title="Glaucoma Detection App", layout="centered")
st.title("👁️ Glaucoma Detection App")

# Debug: show current files
st.text(f"Files in project: {os.listdir()}")

# Load model
try:
    model = load_model("glaucoma_model.h5")
    st.success("Loaded model successfully!")
except Exception as e:
    st.error(f"Couldn't load model: {e}")
    model = None

# Upload and classify
uploaded = st.file_uploader("Upload a retina image", type=["jpg", "jpeg", "png"])
if uploaded and model:
    img = Image.open(uploaded).convert("RGB").resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    pred = model.predict(arr)[0][0]
    st.subheader("Result")
    if pred > 0.5:
        st.error("⚠️ Positive for Glaucoma")
    else:
        st.success("✅ Negative for Glaucoma")
    st.write(f"Confidence Score: {pred:.4f}")
