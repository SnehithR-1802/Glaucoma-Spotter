import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="glaucoma_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_tflite(img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

# Streamlit UI
st.title("Glaucoma Detection (TFLite)")
uploaded = st.file_uploader("Upload a retina image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img = img.resize((224, 224))  # adjust to your model's input size
    st.image(img, caption="Uploaded Image")

    img_arr = np.array(img, dtype=np.float32) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = img_arr.astype(input_details[0]["dtype"])  # match TFLite input dtype

    if img_arr.shape != tuple(input_details[0]["shape"]):
        st.error(f"Input shape mismatch: expected {input_details[0]['shape']}, got {img_arr.shape}")
    else:
        preds = predict_tflite(img_arr)
        index = np.argmax(preds)
        classes = ["Healthy", "Glaucoma"]  # Adjust if needed
        st.write(f"**Prediction:** {classes[index]} ({100 * preds[0][index]:.2f}% confidence)")
