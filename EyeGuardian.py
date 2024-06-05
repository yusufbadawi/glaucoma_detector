import streamlit as st
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from tensorflow import keras

# Disable the deprecation warning for the file uploader encoding
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(page_title="ResNet-50: Glaucoma Detection Assistant", page_icon="https://lh6.googleusercontent.com/proxy/-72rLbBuRJFqDKksT-FpFxD72Ly8lA_zTdDYQRRD9rq7zY4gWiUEt_0AmQzMZ2HSGju2xZMmh81FFVPOLSdzJr_Gxmb6elHrk56mzymeY8RUH9ptsL3xPoIZjjPlhJ-r")

# Load the model with caching to optimize performance
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('my_model2.h5')
    return model

model = load_model()

# Function to preprocess the image and make predictions
def import_and_predict(image_data, model):
    image = ImageOps.fit(image_data, (100, 100), Image.ANTIALIAS)
    image = image.convert('RGB')
    image = np.asarray(image)
    st.image(image, caption='Uploaded Image', use_column_width=False)
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(160deg, #000000 30%, #242434 70%, #346496 100%);
        color: #FFFFFF;
    }
    .sidebar .sidebar-content {
        background-color: #333333;
        color: #FFFFFF;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        cursor: pointer;
        border-radius: 4px;
        margin-top: 10px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        font-size: 1.5em;
        text-align: center;
    }
    .prediction-success {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .prediction-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .footer {
        margin-top: 50px;
        padding: 20px 0;
        background-color: #333333;
        text-align: center;
        font-size: 0.9em;
        color: #FFFFFF;
    }
    a {
        color: #1E90FF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for additional information and navigation
x = "https://www.simpleimageresizer.com/_uploads/photos/e9d53556/unnamed_1_25.png"
st.sidebar.image(x, use_column_width=False)
st.sidebar.title("ResNet-50: Glaucoma Detection Assistant")
st.sidebar.info(
    """
    **By: Ali Ahmed Shawki El Badawi**
    **Supervised By: Dr Mahmoud Khalil**
    
    By simply uploading an image of the eye, users can receive an assessment that helps in identifying the presence of glaucoma, potentially saving vision through early intervention.
    """
)
st.sidebar.write("---")
st.sidebar.write("## About Glaucoma")
st.sidebar.write(
    """
    Glaucoma is a group of eye conditions that damage the optic nerve, often caused by abnormally high pressure in the eye. It's one of the leading causes of blindness for people over the age of 60.
    """
)
st.sidebar.write("For more information, visit [Glaucoma Research Foundation](https://www.glaucoma.org).")

# Main app layout
x = "https://www.simpleimageresizer.com/_uploads/photos/e9d53556/German_University_in_Cairo_Logo_1_10.jpg"
st.image(x, use_column_width=False)
st.title("ResNet-50: Glaucoma Detection Assistant")
st.write(
    """
    Welcome to the Glaucoma Detector! Please upload a fundus image of your eye to receive an assessment.
    """
)

# File uploader for image input
st.write("### Upload your fundus eye image (JPG format)")
file = st.file_uploader("", type=["jpg"])

# Image prediction and result display
if file is None:
    st.warning("You haven't uploaded an image file yet. Please upload a jpg image to proceed.")
else:
    try:
        image = Image.open(file)
        prediction = import_and_predict(image, model)
        pred = prediction[0][0]
        
        if pred > 0.5:
            st.markdown("<div class='prediction-box prediction-success'><strong>Prediction:</strong> Good news! Your eyes are healthy! Keep looking at the bright side, but remember, no direct staring at the sun!</div>", unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown("<div class='prediction-box prediction-error'><strong>Prediction:</strong> Uh-oh, we’ve spotted some signs of glaucoma. Better get it checked out by a professional. Your eyes deserve the best care! No need to panic, but definitely don't turn a blind eye to this.</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing the image: {e}")

# Additional footer or information
st.write("---")
st.write("**Note:** This tool is for my **Bachelor Thesis** & informational purposes only and should not be used as a substitute for professional medical advice.")

# Footer
st.markdown(
    """
    <div class="footer">
        <p>&copy; 2024 Ali Ahmed Shawki El Badawi. All rights reserved.</p>
        <p>For more information, visit the <a href="https://www.glaucoma.org" target="_blank">Glaucoma Research Foundation</a>.</p>
    </div>
    """,
    
    unsafe_allow_html=True
)


