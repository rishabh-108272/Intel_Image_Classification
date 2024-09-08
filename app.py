import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Function to combine the model chunks
def combine_chunks(output_file_path, input_chunks):
    with open(output_file_path, 'wb') as output_file:
        for chunk_file in input_chunks:
            with open(chunk_file, 'rb') as f:
                output_file.write(f.read())
            print(f"Combined chunk: {chunk_file}")

# List of chunk files (you can load these from a GitHub repo or local storage)
chunk_files = [
    './intel_image_cnn_model.keras.part0',
    './intel_image_cnn_model.keras.part1',
    './intel_image_cnn_model.keras.part2',
    './intel_image_cnn_model.keras.part3',
    './intel_image_cnn_model.keras.part4',
    './intel_image_cnn_model.keras.part5',
    './intel_image_cnn_model.keras.part6',
    './intel_image_cnn_model.keras.part7',
    './intel_image_cnn_model.keras.part8',
    './intel_image_cnn_model.keras.part9',
    './intel_image_cnn_model.keras.part10',
]

# Path to the reassembled model
reassembled_model_path = './intel_image_cnn_model_reassembled.keras'

# Combine chunks if the reassembled model doesn't already exist
if not os.path.exists(reassembled_model_path):
    st.write("Reassembling model from chunks...")
    combine_chunks(reassembled_model_path, chunk_files)
    st.success("Model reassembled successfully!")
else:
    st.write("Model already reassembled.")

# Load the Keras model
try:
    model = load_model(reassembled_model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define labels based on your model's output
class_labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Streamlit UI
st.title("Intel Image Classification with Keras Model")
st.write("Upload an image and the model will classify it")
st.caption("The model classifies the image into buildings, forest, glacier, mountain, sea and street.")

# Upload the image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    # Preprocess the image to be ready for model prediction
    img = image.resize((224, 224))  # Adjust to your model's input size
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize image if required by your model
    
    # Predict the image class
    try:
        predictions = model.predict(img)
        confidence = np.max(predictions) * 100  # Get confidence in percentage
        predicted_class = class_labels[np.argmax(predictions)]  # Get predicted class
        
        # Display the prediction
        st.write(f"Predicted Class: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2f}%**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
