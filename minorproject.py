import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import cv2
import io 
import base64 
from flask import Flask  # Add this import statement


st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to download an image
def download_image(image, filename):
    img_path = f"./{filename}"
    image.save(img_path)
    with open(img_path, "rb") as file:
        btn = st.download_button("Download", file.read(), file_name=filename)

# Function to generate a download link
def get_download_link(file_path, link_text):
    with open(file_path, "rb") as file:
        data = file.read()
    b64_data = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{b64_data}" download="{link_text}">{link_text}</a>'
    return href

# Apply steganography
def apply_steganography(image, message):
    # Implementation of steganography goes here
    # Replace with your own steganography algorithm
    return image

# Function to analyze RGB histogram
def analyze_rgb_histogram(image):
    # Convert image to RGB mode if it's not already
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Calculate the histograms for each channel (R, G, B)
    r_histogram = np.histogram(image.getchannel(0), bins=256, range=(0, 256))[0]
    g_histogram = np.histogram(image.getchannel(1), bins=256, range=(0, 256))[0]
    b_histogram = np.histogram(image.getchannel(2), bins=256, range=(0, 256))[0]

    return r_histogram, g_histogram, b_histogram

def main():
    st.title("Streamlit Serverless app") 
    
    st.sidebar.title("Image Editor")

    # Add a file uploader
    uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    # Add an option to take an image from the camera
    if st.sidebar.button("Take Photo"):
        st.title("Camera")
        video_capture = cv2.VideoCapture(0)
        ret, frame = video_capture.read()
        if ret:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            st.image(image, caption="Taken Photo")
            uploaded_file = st.button("Use Photo")
            if uploaded_file:
                image.save("taken_photo.png")

    if uploaded_file is not None:
        # Read the image file
        image = Image.open(io.BytesIO(uploaded_file.read() if hasattr(uploaded_file, 'read') else uploaded_file))
        st.image(image, caption="Uploaded Image")

        # Display image editing options
        st.sidebar.title("Image Editing")
        options = ["Grayscale", "Blur", "Rotate", "Brightness", "Contrast", "Crop"]
        selected_options = st.sidebar.multiselect("Choose image editing options", options)

        edited_image = image.copy()

        for option in selected_options:
            if option == "Grayscale":
                edited_image = edited_image.convert("L")
            elif option == "Blur":
                blur_radius = st.sidebar.slider("Select blur radius", 0, 10, 1)
                edited_image = edited_image.filter(ImageFilter.GaussianBlur(blur_radius))
            elif option == "Rotate":
                angle = st.sidebar.slider("Select rotation angle", -180, 180, 0)
                edited_image = edited_image.rotate(angle)
            elif option == "Brightness":
                brightness_factor = st.sidebar.slider("Select brightness factor", 0.0, 2.0, 1.0, 0.1)
                enhancer = ImageEnhance.Brightness(edited_image)
                edited_image = enhancer.enhance(brightness_factor)
            elif option == "Contrast":
                contrast_factor = st.sidebar.slider("Select contrast factor", 0.0, 2.0, 1.0, 0.1)
                enhancer = ImageEnhance.Contrast(edited_image)
                edited_image = enhancer.enhance(contrast_factor)
            elif option == "Crop":
                st.sidebar.subheader("Crop Image")
                crop_width = st.sidebar.slider("Select crop width", 0, edited_image.width, edited_image.width)
                crop_height = st.sidebar.slider("Select crop height", 0, edited_image.height, edited_image.height)
                left = (edited_image.width - crop_width) / 2
                top = (edited_image.height - crop_height) / 2
                right = (edited_image.width + crop_width) / 2
                bottom = (edited_image.height + crop_height) / 2
                edited_image = edited_image.crop((left, top, right, bottom))

        # Apply steganography
        message = st.sidebar.text_input("Enter a message for steganography")
        if st.sidebar.button("Apply Steganography"):
            steganographic_image = apply_steganography(edited_image, message)
            st.sidebar.subheader("Steganographic Image")
            st.sidebar.image(steganographic_image)

        # Show the edited image
        st.subheader("Edited Image")
        st.image(edited_image)

        # Analyze RGB histogram
        if st.sidebar.button("Analyze RGB Histogram"):
            r_histogram_orig, g_histogram_orig, b_histogram_orig = analyze_rgb_histogram(image)
            r_histogram_edited, g_histogram_edited, b_histogram_edited = analyze_rgb_histogram(edited_image)

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(range(256), r_histogram_orig, color='red', label='Red')
            plt.plot(range(256), g_histogram_orig, color='green', label='Green')
            plt.plot(range(256), b_histogram_orig, color='blue', label='Blue')
            plt.title('Original Image RGB Histograms')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(range(256), r_histogram_edited, color='red', label='Red')
            plt.plot(range(256), g_histogram_edited, color='green', label='Green')
            plt.plot(range(256), b_histogram_edited, color='blue', label='Blue')
            plt.title('Edited Image RGB Histograms')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.legend()

            st.pyplot()

        # Allow the user to download the image
        if st.sidebar.button("Download Image"):
            download_image(image, "original_image.jpg")
            download_image(edited_image, "edited_image.jpg")
            st.sidebar.markdown(get_download_link("original_image.jpg", "Original_Image.jpg"), unsafe_allow_html=True)
            st.sidebar.markdown(get_download_link("edited_image.jpg", "Edited_Image.jpg"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
