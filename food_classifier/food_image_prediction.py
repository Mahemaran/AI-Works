import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the pre-trained model
model = load_model(r"C:\Users\DELL\PycharmProjects\pythonProject\Practice\images\food_classifier.h5")

# List of food categories (or use your own)
food_categories = ['Boiled Egg', 'Green Dal', 'Omelets', 'White Rice']  # Modify this list with your food classes

def load_and_predict(img):
    img = img.resize((224, 224))  # Resize the image to match model input
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)  # Get the index of the highest probability
    return food_categories[predicted_class]

# Streamlit UI setup
st.title("Food Image Classifier")
st.write("Take a photo and let the model predict the food name.")

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("Classifying...")

    # Make prediction
    predicted_food = load_and_predict(img)
    st.write(f"**Prediction**: {predicted_food}")

# Option for webcam photo
# st.write("Or use the webcam to capture a food image:")

# camera_input = st.camera_input("Take a photo")

# if camera_input is not None:
#     img = Image.open(camera_input)
#     st.image(img, caption="Captured Image", use_column_width=True)
#     st.write("Classifying...")
#
#     # Make prediction
#     predicted_food = load_and_predict(img)
#     st.write(f"Prediction: {predicted_food}")
