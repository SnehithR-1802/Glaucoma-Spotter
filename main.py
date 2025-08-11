import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load your model
model = load_model("glaucoma_model.h5")

st.title("👁 Glaucoma Detection App")

# Upload image
uploaded_file = st.file_uploader("Upload a retina image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize to match training input shape
    img = img.resize((224, 224))  # MobileNetV2 trained size

    # Convert to array
    img_arr = np.array(img, dtype=np.float32)

    # Expand dims & preprocess exactly as in training
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)

    # Predict
    prediction = model.predict(img_arr)
    st.write("🔍 Raw model output:", prediction)

    # Interpret prediction
    class_names = ["Healthy", "Glaucoma"]
    predicted_class = class_names[np.argmax(prediction)]
    st.write(f"**Prediction:** {predicted_class}")
