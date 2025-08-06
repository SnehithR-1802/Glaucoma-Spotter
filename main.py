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

# ✅ Streamlit UI
st.title("Glaucoma Detection")

uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))  # Match your training size
    img_arr = np.array(image) / 255.0

    preds = predict_tflite(img_arr)  # ✅ This line should now work

    # ✅ DEBUG: show raw predictions
    st.write(f"Raw model output: {preds}")

    classes = ["Glaucoma", "Healthy"]  # Try reversing if predictions are wrong
    index = np.argmax(preds)
    st.write(f"**Prediction:** {classes[index]} ({100 * preds[index]:.2f}% confidence)")
