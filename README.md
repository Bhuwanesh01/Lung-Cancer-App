# Lung-Cancer-App
An app designed to predict lung cancer by uploading ct scan images.

# Lung Cancer Detection AI

An AI-powered web application for detecting lung cancer from CT scan images using deep learning.  
Built with **TensorFlow, Streamlit, and Grad-CAM explainability**.

---

##  Features

-  **Lung Cancer Detection**
  - Classifies CT scan images as **Normal** or **Cancer**
  - Displays prediction confidence

-  **AI Prescription**
  - Generates basic medical guidance based on prediction

-  **Grad-CAM Visualization**
  - Highlights regions of the image influencing the model

-  **Probability Graph**
  - Visual comparison of Normal vs Cancer prediction

-  **PDF Report Generation**
  - Download structured medical-style report
  - Includes patient details, result, confidence, and prescription

-  **Patient History**
  - Stores previous predictions within session

-  **Patient Input Panel**
  - Name, Age, Gender, Description
  - Adjustable detection threshold

---

## Tech Stack

- **Frontend & App**: Streamlit  
- **Model**: TensorFlow / Keras (MobileNetV2 - Transfer Learning)  
- **Visualization**: Matplotlib, OpenCV  
- **Image Processing**: Pillow  
- **PDF Generation**: ReportLab  

---

## 📁 Project Structure
