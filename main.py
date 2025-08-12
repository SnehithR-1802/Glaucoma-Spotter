from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.applications.resnet import preprocess_input
import numpy as np

# Load your model
model = load_model("glaucoma_model.h5")

IMG_SIZE = (224, 224)  # Must match training

def predict_image(img_path):
    # Load and resize
    img = image.load_img(img_path, target_size=IMG_SIZE)
    
    # Convert to array
    img_array = image.img_to_array(img)
    
    # Add batch dimension
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Preprocess EXACTLY like training
    img_preprocessed = preprocess_input(img_batch)
    
    # Predict
    prediction = model.predict(img_preprocessed)
    
    # Convert to class label
    class_idx = np.argmax(prediction, axis=1)[0]
    class_labels = ["Glaucoma", "Healthy"]  # Replace with your exact order
    return class_labels[class_idx], prediction

# Example:
label, raw_pred = predict_image("test_image.jpg")
print(f"Prediction: {label}")
print(f"Raw output: {raw_pred}")
