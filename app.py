import io

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ---------------- Page config ----------------
st.set_page_config(page_title="RespiDetect AI", page_icon="🫁", layout="centered")

st.title("🫁 RespiDetect: Respiratory Disease Screening")
st.markdown("""
AI-powered respiratory disease detection.  
🎤 **Record live** or 📁 **upload** a .wav file for instant analysis.
""")


# ---------------- Model loading (cached) ----------------
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.load_state_dict(
        torch.load("models/respidetect_mobilenet.pth", map_location=device)
    )
    model.eval()
    return model, device


model, device = load_model()

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------- Audio -> spectrogram -> tensor ----------------
def process_audio_bytes(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=44100, mono=True)

    if y.size == 0 or np.max(np.abs(y)) < 1e-6:
        raise ValueError("Recording is empty or silent. Please record again.")

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(mel_spec_db, aspect="auto", cmap="magma")
    ax.axis("off")
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((224, 224))

    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    return tensor, fig


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write("**Model:** MobileNetV2")
    st.write("**Test Accuracy:** 84.24%")
    st.write("**Healthy Recall:** 88.89%")
    st.divider()
    st.caption("⚠️ Screening tool only. Consult a doctor for diagnosis.")


# ---------------- Input tabs ----------------
tab1, tab2 = st.tabs(["🎤 Live Record", "📁 Upload File"])

audio_bytes = None

with tab1:
    st.subheader("Live Microphone Recording")
    st.info(
        "Click the mic icon, breathe normally for 5-10 seconds, "
        "then click stop."
    )

    # Built-in Streamlit recorder: returns a WAV file directly
    recorded = st.audio_input("🎤 Record your breathing")

    if recorded is not None:
        audio_bytes = recorded.getvalue()
        st.success("✅ Recording captured! Processing...")

with tab2:
    st.subheader("Upload Audio File")
    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        st.success("✅ File uploaded! Processing...")


# ---------------- Prediction ----------------
if audio_bytes is not None:
    try:
        tensor, spec_fig = process_audio_bytes(audio_bytes)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.softmax(output, dim=1)[0]
            pred_class = torch.argmax(prob).item()
            confidence = prob[pred_class].item() * 100

        col1, col2 = st.columns(2)
        with col1:
            if pred_class == 0:
                st.success(f"✅ **Healthy** ({confidence:.1f}%)")
            else:
                st.error(f"⚠️ **Potential Abnormality** ({confidence:.1f}%)")
        with col2:
            st.metric("Confidence", f"{confidence:.1f}%")

        st.subheader("📊 Spectrogram Analysis")
        st.pyplot(spec_fig)
        plt.close(spec_fig)
        st.audio(audio_bytes, format="audio/wav")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👆 Use the tabs above to record or upload audio.")
