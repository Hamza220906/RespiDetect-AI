import io
from torchvision import transforms
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile


st.set_page_config(page_title="RespiDetect AI", page_icon="🫁", layout="centered")


st.title("🫁 RespiDetect: Respiratory Disease Screening")
st.markdown("""
AI-powered respiratory disease detection using smartphone audio recordings.  
Upload an audio file (.wav) to analyze breathing patterns and detect potential abnormalities.
""")


@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load("models/respidetect_mobilenet.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()


def process_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=44100)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
   
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(mel_spec_db, aspect='auto', cmap='magma')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((224, 224))
    
   
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor, fig


with st.sidebar:
    st.header("ℹ️ About")
    st.write("**Model:** MobileNetV2")
    st.write("**Test Accuracy:** 84.24%")
    st.write("**Healthy Recall:** 88.89%")
    st.write("**Dataset:** ICBHI Challenge")
    st.divider()
    st.caption("⚠️ This is a screening tool, not a medical diagnosis. Always consult a doctor.")


uploaded_file = st.file_uploader("🎤 Upload Audio File (.wav)", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    try:
        st.subheader(" Analysis Result")
        
       
        tensor, spec_fig = process_audio(tmp_path)
        
        with torch.no_grad():
            output = model(tensor)
            prob = torch.softmax(output, dim=1)[0]
            pred_class = torch.argmax(prob).item()
            confidence = prob[pred_class].item() * 100
        
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if pred_class == 0:
                st.success(f"✅ **Healthy** ({confidence:.1f}%)")
                st.info("No significant respiratory abnormalities detected.")
            else:
                st.error(f"⚠️ **Potential Abnormality Detected** ({confidence:.1f}%)")
                st.warning("Consult a healthcare professional for further evaluation.")
        
        with col2:
            st.metric("Confidence", f"{confidence:.1f}%")
        
        
        st.subheader(" Spectrogram Analysis")
        st.pyplot(spec_fig)
        
        
        st.audio(tmp_path, format="audio/wav")
        
    except Exception as e:
        st.error(f"❌ Error processing audio: {str(e)}")
    finally:
        os.unlink(tmp_path)
else:
    st.info(" Upload a .wav audio file above to begin analysis.")