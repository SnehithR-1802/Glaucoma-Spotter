import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.applications.resnet import preprocess_input
import numpy as np
from PIL import Image
import tempfile

# Load model once
model = load_model("glaucoma_model.h5")

IMG_SIZE = (224, 224)
class_labels = ["Glaucoma", "Healthy"]  # Replace with your actual training order

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    prediction = model.predict(img_preprocessed)
    class_idx = np.argmax(prediction, axis=1)[0]
    return class_labels[class_idx], prediction

# Streamlit UI
st.title("Glaucoma Detector")

uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        img = Image.open(uploaded_file).convert("RGB")
        img.save(tmp.name, format="JPEG")
        label, raw_pred = predict_image(tmp.name)

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write(f"**Prediction:** {label}")
    st.write(f"**Raw probabilities:** {raw_pred}")
