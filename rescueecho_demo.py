import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from streamlit_lottie import st_lottie
import pyttsx3
import threading
import streamlit.components.v1 as components

st.set_page_config(page_title="RescueEcho Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        body {
            background-color: #0f1116;
            color: #ffffff;
        }
        .css-1d391kg, .css-1v0mbdj, .css-ffhzg2 {
            background-color: #1e1e1e;
            color: white;
        }
        .stButton>button {
            background-color: #00acc1;
            color: white;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #00838f;
        }
    </style>
""", unsafe_allow_html=True)

def speak(text):
    def run_speech():
        try:
            local_engine = pyttsx3.init()
            local_engine.setProperty('rate', 150)
            local_engine.setProperty('volume', 1.0)
            local_engine.say(text)
            local_engine.runAndWait()
        except Exception as e:
            st.error(f"TTS Error: {e}")
            print("TTS Error:", e)
    threading.Thread(target=run_speech).start()

def speak_browser(text):
    js_code = f"""
        <script>
            var msg = new SpeechSynthesisUtterance("{text}");
            msg.rate = 1;
            msg.pitch = 1;
            msg.lang = "en-US";
            window.speechSynthesis.speak(msg);
        </script>
    """
    components.html(js_code, height=0)

def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Animation Load Error: {e}")
        return None

lottie_radar = load_lottiefile("radar_animation.json")

st.sidebar.title("RescueEcho Info")
st.sidebar.markdown("""
**Created by:** Srishti Chamoli  
**Stack:** TensorFlow | Streamlit | CNN | Radar Spectrograms  
**Objective:** Post-disaster survivor detection using radar motion analysis  
**Prototype:** Functional, AI-ready (Hardware in progress)
""")

left, right = st.columns([2, 1])
with left:
    st.title("RescueEcho: AI Radar for Disaster Response")
    st.markdown("""
    <div style='font-size:18px;'>
        Leveraging radar spectrograms & deep learning to detect human motion patterns in critical post-disaster environments.
    </div>
    """, unsafe_allow_html=True)
with right:
    if lottie_radar:
        st_lottie(lottie_radar, height=250, key="radar")
    else:
        st.warning("Radar animation failed to load. Check file path or format.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("RescueEcho_model (1).h5")

model = load_model()
class_names = sorted([
    'Cane', 'Crawling', 'Creeping', 'Crutch', 'Fall Sideways',
    'Falling Front', 'HO', 'Limping', 'Running', 'Tripod',
    'Walker', 'Walking', 'Wheeled Chair'
])

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0).astype(np.float32)

def predict_and_display(image):
    input_tensor = preprocess_image(image)
    prediction = model.predict(input_tensor)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    speak(f"{predicted_class} motion detected with {confidence:.2f} percent confidence.")
    speak_browser(f"{predicted_class} motion detected with {confidence:.2f} percent confidence.")

    st.markdown(f"""
        <div style='padding: 20px; background-color: #263238; border-left: 6px solid #00acc1;'>
            <h4 style='color:#00e5ff; font-weight:bold;'><span style='font-size: 26px'>{predicted_class.upper()} DETECTED</span></h4>
            <p style='margin:0; font-size:18px;'>Confidence: <b>{confidence:.2f}%</b></p>
        </div>
    """, unsafe_allow_html=True)

    fig_width = 16
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    bars = sns.barplot(x=class_names, y=prediction, palette="viridis", ax=ax)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10, color='white')
    ax.set_ylabel("Confidence", fontsize=12, color='white')
    ax.set_title("Prediction Confidence", fontsize=16, fontweight='bold', color='white')
    ax.set_facecolor("#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")

    for bar in bars.patches:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', (bar.get_x() + bar.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')

    ax.tick_params(colors='white')
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)


st.subheader("Real-Time Radar Spectrogram Classification")
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload a Radar Spectrogram (JPG/PNG)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Spectrogram Preview", use_container_width=True)
        predict_and_display(image)

with col2:
    st.markdown("""#### System Status: ONLINE  
    Model loaded successfully. Ready to classify.  
    """)
    st.success("Radar Classifier Active")

    st.markdown("""#### Grad-CAM (Coming Soon)
    Visual explanation of motion regions contributing most to prediction.
    """)
    st.info("Enable Grad-CAM in next model version.")

with st.expander("Try Preloaded Spectrograms"):
    sample = st.selectbox("Select a motion", ["Walking", "Running", "Fall Sideways"])
    sample_path = f"samples/{sample.lower().replace(' ', '_')}.png"
    if os.path.exists(sample_path):
        img = Image.open(sample_path)
        st.image(img, caption=f"Sample: {sample}", use_container_width=True)
        predict_and_display(img)
    else:
        st.warning("Sample not found.")

with st.expander("Model Performance Insights"):
    st.markdown("Training & Validation overview of RescueEcho's model")
    if os.path.exists("plots/accuracy_loss.png"):
        st.image("plots/accuracy_loss.png", caption="Accuracy & Loss", use_container_width=True)
    if os.path.exists("plots/confusion_matrix.png"):
        st.image("plots/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)

with st.expander("How RescueEcho Works"):
    st.image("pipeline/image.png", caption="Radar-to-Rescue Pipeline", use_container_width=True)
    st.markdown("""
    - **Radar sensors** detect motion using micro-Doppler shifts.
    - Signals are converted to **time-frequency spectrograms**.
    - A trained CNN (MobileNetV2) performs real-time classification.
    - Use-case: locate survivors under debris or poor visibility conditions.
    """)