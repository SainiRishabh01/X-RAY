# Chest X-Ray Image Analysis

## Overview

This Streamlit app analyzes chest X-ray images to detect potential abnormalities using a pre-trained DenseNet121 model. It provides a user-friendly interface for uploading and processing images, and it visualizes predictions with Grad-CAM.

Check out the live Streamlit App [here](https://chestxrayanalysis.streamlit.app).

## Features

- **Image Upload**: Accepts chest X-ray images in JPG, JPEG, or PNG format.
- **Abnormality Detection**: Uses a machine learning model to predict the probability of 14 different conditions.
- **Grad-CAM Visualization**: Highlights areas of the image most relevant to the model's predictions.
- **Interactive Interface**: Built with Streamlit for easy and intuitive use.

## Important Notes

- This tool is for educational purposes only.
- It is not a substitute for professional medical advice or diagnosis.
- Ensure you upload only chest X-ray images for accurate results.

## Instructions

1. **Upload a Chest X-Ray Image**: Click on the "Upload a chest X-ray image" button and select an image file.
2. **View the Results**: The app will display the uploaded image, prediction results, and Grad-CAM visualizations.
3. **Explore Further**: Visit the portfolio and source code repository for more projects and insights.

## How to Run Locally

1. Clone this repository:

    ```bash
    git clone https://github.com/kimnguyen2002/Chest-X-Ray-Image-Analysis.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Chest-X-Ray-Image-Analysis
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

## Model Details

The model is based on DenseNet121 and is trained to identify 14 conditions:

- Cardiomegaly
- Emphysema
- Effusion
- Hernia
- Infiltration
- Mass
- Nodule
- Atelectasis
- Pneumothorax
- Pleural Thickening
- Pneumonia
- Fibrosis
- Edema
- Consolidation

## License

This project is licensed under the MIT License.

Feel free to adjust the structure to fit your actual project layout!
