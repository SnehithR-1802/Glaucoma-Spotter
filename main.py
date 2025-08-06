import streamlit as st
import numpy as np
from PIL import Image
import tensorflow.lite as tflite

# Load TFLite model
interpreter = tflite.Interpreter(model_path="glaucoma_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_tflite(img_array):
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return output[0]

# Streamlit UI
st.title("Glaucoma Detection (TFLite)")
uploaded = st.file_uploader("Upload a retina image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB").resize((224, 224))
    st.image(img, caption="Uploaded Image")
    img_arr = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

preds = predict_tflite(img_arr)

classes = ["Healthy", "Glaucoma"]
index = np.argmax(preds)
st.write(f"Raw model output: {preds}")
st.write(f"**Prediction:** {classes[index]} ({100 * preds[index]:.2f}% confidence)")

