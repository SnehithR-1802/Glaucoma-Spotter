import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ✅ Define the prediction function (make sure this is ABOVE where you call it)
def predict_tflite(img_array):
    interpreter = tf.lite.Interpreter(model_path="glaucoma_model.tflite")  # or .h5 if not TFLite
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]  # Return the prediction array

# Streamlit UI
st.title("Glaucoma Detection (TFLite)")
uploaded = st.file_uploader("Upload a retina image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img = img.resize((224, 224))  # Adjust if needed based on your model
    st.image(img, caption="Uploaded Image")

    img_arr = np.array(img, dtype=np.float32) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    # Ensure correct dtype and shape
    expected_shape = input_details[0]["shape"]
    expected_dtype = input_details[0]["dtype"]

    img_arr = img_arr.astype(expected_dtype)

    if img_arr.shape != tuple(expected_shape):
        st.error(f"Input shape mismatch: expected {expected_shape}, got {img_arr.shape}")
    else:
        preds = predict_tflite(img_arr)
        index = np.argmax(preds)
        classes = ["Healthy", "Glaucoma"]  # Reverse if needed
        st.write(f"**Prediction:** {classes[index]} ({100 * preds[index]:.2f}% confidence)")
