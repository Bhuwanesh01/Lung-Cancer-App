import os
import io
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import cv2


# PDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ---------------- PAGE ----------------
st.set_page_config(page_title="Lung Cancer Detection AI", layout="wide")
st.title("🧠 Lung Cancer Detection AI")

# ---------------- MODEL ----------------
MODEL_PATH = "model.keras"

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.stop()
    return load_model(MODEL_PATH, compile=False)

model = load_my_model()

# ---------------- PREPROCESS ----------------
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img).astype("float32") / 255.0

    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    return np.expand_dims(img, axis=0)

# ---------------- GRAD-CAM ----------------
def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def get_gradcam(model, img_array):
    try:
        last_conv = get_last_conv_layer_name(model)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv).output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_output, preds = grad_model(img_array)
            loss = preds[:, 0]

        grads = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
        conv_output = conv_output[0]

        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8
        return heatmap
    except:
        return None

def overlay_heatmap(heatmap, image):
    h, w = image.height, image.width
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img = np.array(image)
    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# ---------------- PDF ----------------
def generate_pdf(name, age, gender, result, confidence, prescription):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 760, "Lung Cancer Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(100, 720, f"Name: {name}")
    c.drawString(100, 700, f"Age: {age}")
    c.drawString(100, 680, f"Gender: {gender}")

    c.drawString(100, 650, f"Result: {result}")
    c.drawString(100, 630, f"Confidence: {confidence:.2f}")

    # AI Prescription
    y = 600
    c.setFont("Helvetica-Bold", 13)
    c.drawString(100, y, "AI Prescription:")

    c.setFont("Helvetica", 10)
    y -= 20

    for line in prescription.split("\n"):
        c.drawString(100, y, line.strip())
        y -= 15

    c.save()
    buffer.seek(0)
    return buffer
# ---------------- SIDEBAR ----------------
st.sidebar.header("👤 Patient Details")

name = st.sidebar.text_input("Name", "Ayush")
age = st.sidebar.number_input("Age", 1, 120, 20)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

# ✅ NEW FIELD
description = st.sidebar.text_area("Patient Description", "Enter symptoms / notes...")

threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.5)

# ---------------- HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- UPLOAD ----------------
files = st.file_uploader(
    "Upload CT Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# AI PRESCRIPTION FUNCTION

def generate_prescription(result, confidence):

    confidence = round(float(confidence), 2)

    if result == "Normal":
        return f"""
### ✅ Low Risk Detected (Confidence: {confidence})

**Recommended Actions:**
- Maintain a healthy lifestyle  
- Regular checkups every 6–12 months  
- Avoid smoking and air pollution  
- Practice breathing exercises  

✔ No immediate concern, but stay cautious.
"""

    else:
        return f"""
### ⚠ High Risk Detected (Confidence: {confidence})

**Recommended Actions:**
- Consult a pulmonologist immediately  
- Get CT scan / biopsy confirmation  
- Avoid smoking and polluted environments  
- Maintain proper nutrition and rest  

⚠ Early medical attention is strongly advised.
"""


# ---------------- MAIN ----------------
i = 1
if files:
    for file in files:
        
        i=i+1
        st.markdown("---")
        st.markdown(f"### 📁 {file.name}")

        image = Image.open(file).convert("RGB")
        img_array = preprocess_image(image)

        col1, col2, col3 = st.columns([1,1,1])

        # IMAGE
        with col1:
            st.image(image, caption="Original", width=300)

        # PREDICTION
        pred = float(model.predict(img_array, verbose=0)[0][0])
        cancer_prob = pred
        normal_prob = 1 - pred

        if cancer_prob > threshold:
            result = "Cancer"
            confidence = cancer_prob
        else:
            result = "Normal"
            confidence = normal_prob

        with col2:
            st.subheader("Prediction")

            if result == "Cancer":
                st.error("⚠ Cancer Detected")
            else:
                st.success("✅ Normal")

            st.write(f"Confidence: {confidence:.2f}")

            # SMALL GRAPH
            fig, ax = plt.subplots(figsize=(3,3))
            ax.bar(["Normal","Cancer"], [normal_prob, cancer_prob])
            ax.set_ylabel("Prob")
            st.pyplot(fig)

        # GRADCAM
        with col3:
            st.subheader("Grad-CAM")

            heatmap = get_gradcam(model, img_array)
            if heatmap is not None:
                cam = overlay_heatmap(heatmap, image)
                st.image(cam, width=300)
            else:
                st.info("Grad-CAM not available")

        # SHOW AI PRESCRIPTION IN UI
        st.markdown("---")
        # ===== AI PRESCRIPTION =====
        prescription = generate_prescription(result, confidence)

        st.subheader("🧠 AI Prescription")

        st.markdown(f"""
        <div style="
            padding:15px;
            border-radius:10px;
            background-color:#0f172a;
            border-left:5px solid {'#dc2626' if result=='Cancer' else '#16a34a'};
        ">
        <pre style="white-space:pre-wrap; color:white; font-size:14px;">
        {prescription}
        
        """, unsafe_allow_html=True)

        # HISTORY
        case_id = f"{file.name}_{round(confidence,3)}"

        if "history_ids" not in st.session_state:
            st.session_state.history_ids = set()

        if case_id not in st.session_state.history_ids:

            st.session_state.history_ids.add(case_id)

            if "history" not in st.session_state:
                st.session_state.history = []

            st.session_state.history.append({
                "name": name,
                "age": age,
                "gender": gender,
                "result": result,
                "description": description,
                "confidence": round(float(confidence), 3)
            })
        
        # # ===== AI PRESCRIPTION =====
        # st.subheader("🧠 AI Prescription")
        # st.markdown(prescription)

        # PDF
        pdf = generate_pdf(name, age, gender, result, confidence, prescription)
        st.markdown("---")

        st.download_button(
            label="📄 Download Report",
            data=pdf,
            file_name=f"{name}_report_{i}.pdf",
            mime="application/pdf",
            key=f"download_{i}"
        )


# ---------------- HISTORY DISPLAY ----------------
st.markdown("---")
st.subheader("📜 History")

if not st.session_state.history:
    st.info("No history yet")
else:
    for i, item in enumerate(reversed(st.session_state.history), start=1):

        color = "#16a34a" if item["result"] == "Normal" else "#dc2626"
        icon = "🟢" if item["result"] == "Normal" else "🔴"

        st.markdown(f"""
        <div style="
            padding:15px;
            border-radius:10px;
            margin-bottom:10px;
            background-color:#111827;
            border-left:5px solid {color};
        ">
            <b>Case {i}</b><br>
            👤 <b>{item['name']}</b> | Age: {item['age']} | Gender: {item['gender']}<br>
            📝 {item['description']}<br>
            {icon} <b>{item['result']}</b><br>
            📊 Confidence: {item['confidence']:.2f}
        </div>
        """, unsafe_allow_html=True)