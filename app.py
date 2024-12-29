import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing import image
from keras.models import Model
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense
import matplotlib.pyplot as plt
import pandas as pd

# Function to rebuild the model and load weights
@st.cache_resource
def load_trained_model():
    base_model = DenseNet121(weights=None, include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(14, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights('pretrained_model.h5')
    return model

# Function to preprocess uploaded image
def preprocess_image(img):
    img = cv2.resize(img, (320, 320))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to generate GradCAM visualization
def grad_cam(input_model, image, cls, layer_name='bn'):
    grad_model = tf.keras.models.Model(
        inputs=[input_model.inputs], 
        outputs=[input_model.get_layer(layer_name).output, input_model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, cls]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    conv_outputs = conv_outputs * pooled_grads
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1e-10
    heatmap = cv2.resize(heatmap, (320, 320))
    return heatmap

# Function to check if the image is likely a chest X-ray
def is_chest_xray(img):
    # Check if the image is grayscale (typical for X-rays)
    if len(img.shape) == 2 or img.shape[2] == 1:
        return True

    # Check if the image contains color
    if not is_grayscale(img):
        return False

    # Check the aspect ratio of the image (Chest X-rays are often rectangular with common aspect ratios)
    aspect_ratio = img.shape[1] / img.shape[0]
    if 0.7 < aspect_ratio < 1.5:  # Typical aspect ratio range for medical images
        return True

    return False

# Function to check if an image is grayscale
def is_grayscale(img):
    # Calculate the difference between the RGB channels
    if len(img.shape) == 3:
        b, g, r = cv2.split(img)  # Split into Blue, Green, Red channels
        if np.allclose(b, g) and np.allclose(g, r):
            return True
    return False

# Streamlit App
st.set_page_config(page_title="Chest X-Ray Image Analysis", layout="centered")
st.title("🩺 Chest X-Ray Image Analysis - Detect Abnormalities")

# Instructions and Warning
st.markdown("""
## Instructions
- Upload a chest X-ray image in JPG, JPEG, or PNG format.
- The app will analyze the image for potential abnormalities.

### Important Notes
- This tool is for educational purposes only.
- It is not a substitute for professional medical advice or diagnosis.
- Ensure you upload only chest X-ray images for accurate results.
""")

# File upload
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Convert the file to an image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Check if the uploaded image contains color (X-rays are typically grayscale)
        if not is_grayscale(img):
            st.warning("The uploaded image appears to contain color, which is unusual for chest X-rays. Please upload a grayscale chest X-ray image.")

        # Check if the uploaded image is likely to be a chest X-ray
        if not is_chest_xray(img):
            st.error("The uploaded image does not appear to be a chest X-ray. Please upload a proper chest X-ray image.")
        else:
            # Display image
            st.image(img, channels="BGR", caption="Uploaded Image", use_column_width='bool', width=400)

            # Preprocess image
            preprocessed_img = preprocess_image(img)

            # Load model
            model = load_trained_model()

            # Make prediction
            predictions = model.predict(preprocessed_img)

            # Labels from the original dataset
            labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass',
                      'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening', 'Pneumonia',
                      'Fibrosis', 'Edema', 'Consolidation']

            # Display prediction results
            st.subheader("Prediction Results")
            results_df = pd.DataFrame({"Condition": labels, "Probability": predictions[0]})

            # Apply consistent highlighting
            styled_df = results_df.style.apply(lambda x: ['background-color: yellow' if v == x.max() else '' for v in x], subset=['Probability'])

            st.dataframe(styled_df, width=700)

            # Grad-CAM visualization for the top predicted labels
            top_labels = np.argsort(predictions[0])[::-1][:4]

            st.subheader("Grad-CAM Visualization")
            fig, axes = plt.subplots(1, 5, figsize=(20, 10))
            axes[0].imshow(img)
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            for i, idx in enumerate(top_labels):
                gradcam = grad_cam(model, preprocessed_img, idx)
                axes[i+1].imshow(img)
                axes[i+1].imshow(gradcam, cmap='jet', alpha=0.5)
                axes[i+1].set_title(f"{labels[idx]}: {predictions[0][idx]:.3f}")
                axes[i+1].axis('off')

            st.pyplot(fig)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}. Please ensure that you upload a valid chest X-ray image.")

else:
    st.info("Please upload a chest X-ray image in JPG, JPEG, or PNG format.")

st.markdown("""---""")
st.write("""
### Portfolio
For more projects and insights, please visit my [portfolio](https://github.com/SainiRishabh01/).

### Code Repository
You can find the complete code for this application in my GitHub repository [here](https://github.com/SainiRishabh01).
""")
