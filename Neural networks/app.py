
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Set page layout
st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")
st.title("MNIST Digit Classifier")
st.write("Upload a 28x28 grayscale image of a handwritten digit (0–9) to get a prediction.")

# Load the saved model (.h5 format)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_digit_classifier.h5")
    return model

model = load_model()

# Function to preprocess uploaded image
def preprocess_image(img):
    img = img.convert("L")                # Convert to grayscale
    img = img.resize((28, 28))            # Resize to 28x28
    img = ImageOps.invert(img)            # Invert colors (white digits on black)
    img_array = np.array(img) / 255.0     # Normalize
    img_array = img_array.reshape(1, 784) # Flatten to 784
    return img_array

# File uploader
uploaded_file = st.file_uploader("📤 Upload a 28x28 grayscale digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=150)

    # Preprocess and Predict
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    predicted_label = np.argmax(prediction)

    st.markdown(f"# Predicted Digit: `{predicted_label}`")
    st.markdown("#Prediction Probabilities")
    st.bar_chart(prediction[0])
else:
    st.info("Please upload an image to begin.")
