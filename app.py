import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pickle
import cv2

# Load the pre-trained model
with open("models/forest", "rb") as f:
    model = pickle.load(f)

st.title("Draw and Analyze")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="#ffffff",  # Fixed fill color with some opacity
    stroke_width=10,
    stroke_color="#000000",
    background_color="#ffffff",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Button to process the drawing
if st.button("Analyze Drawing"):
    if canvas_result.image_data is not None:
        # Convert canvas image to a numpy array
        img_data = canvas_result.image_data

        # Convert the image to grayscale
        gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        gray = 255 - gray

        # Resize image to the size expected by your model
        img_resized = cv2.resize(gray, (8, 8))

        # Flatten the image to feed into the model
        img_flatten = img_resized.flatten().reshape(1, -1)

        # Make a prediction
        prediction = model.predict(img_flatten)

        # Display the prediction
        st.write(f"Prediction: Number {prediction[0]}")
    else:
        st.write("Please draw something to analyze.")
