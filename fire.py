import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# ===============================
# Load Trained Model
# ===============================
@st.cache_resource
def load_fire_model():
    return load_model("BEST_MODEL.h5")

model = load_fire_model()
IMG_SIZE = (124, 124)

# ===============================
# Preprocessing Function
# ===============================
def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, IMG_SIZE)
    frame_resized = frame_resized.astype("float32") / 255.0
    return np.expand_dims(frame_resized, axis=0)

# ===============================
# Page Config
# ===============================
st.set_page_config(page_title="Fire Detection System", page_icon="🔥", layout="centered")

st.title("🔥 Fire Detection System")
st.markdown(
    "<h4 style='color:gray;'>AI-powered real-time fire detection using CNN</h4>",
    unsafe_allow_html=True
)

# Tabs for Upload & Camera
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Real-Time Camera"])

# ===============================
# Tab 1: Image Upload
# ===============================
with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_array = preprocess_image(img)
        prediction = model.predict(img_array)[0].item()

        if prediction < 0.5:
            st.error(f"🔥 FIRE DETECTED! ({prediction*100:.2f}%)")
        else:
            st.success(f"✅ No Fire Detected ({(1-prediction)*100:.2f}%)")

# ===============================
# Tab 2: Real-Time Camera
# ===============================
with tab2:
    st.write("Enable your webcam for live fire detection.")
    camera_input = st.camera_input("Take a snapshot")

    if camera_input is not None:
        img = Image.open(camera_input)
        st.image(img, caption="Captured Frame", use_column_width=True)

        img_array = preprocess_image(img)
        prediction = model.predict(img_array)[0].item()

        if prediction < 0.5:
            st.error(f"🔥 FIRE DETECTED! ({prediction*100:.2f}%)")
        else:
            st.success(f"✅ No Fire Detected ({(1-prediction)*100:.2f}%)")
