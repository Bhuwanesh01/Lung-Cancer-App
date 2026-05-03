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
lung-cancer-app/
│
├── app.py # Main Streamlit application
├── model.keras # Trained deep learning model
├── requirements.txt # Dependencies
└── README.md


---

## ⚙️ Installation

### 1. Clone the repository


git clone https://github.com/your-username/lung-cancer-app.git
cd lung-cancer-app

2. Install dependencies
pip install -r requirements.txt

3. Run the app
streamlit run app.py

🌐 Deployment

This app is designed to be deployed on Streamlit Cloud.

Steps:
Push your project to GitHub
Go to https://streamlit.io/cloud
Click New App
Select your repo and app.py
Deploy 🚀
📌 Notes
Ensure model.keras is present in the root directory
Use tensorflow-cpu for smoother cloud deployment
Avoid large model sizes (>100MB) unless using Git LFS
⚠ Disclaimer

This project is for educational and research purposes only.
It is not a medical diagnostic tool and should not replace professional medical advice.

🙌 Future Improvements
Multi-class classification (benign / malignant)
Database integration for patient records
Authentication system
Batch report download (ZIP)
Advanced explainability (SHAP, LIME)
👨‍💻 Author

Bhuwanesh
Computer Science Engineering | AI/ML Enthusiast

⭐ If you like this project

Give it a star ⭐ on GitHub!

