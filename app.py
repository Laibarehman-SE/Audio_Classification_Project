import streamlit as st
import numpy as np
import tensorflow as tf
import librosa

st.set_page_config(page_title="AcoustiSense AI", page_icon="🎵")

st.title("🎵 AcoustiSense AI - Audio Classification")

# 1. Load Model and Classes
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model("audio_classification_cnn.h5")
    classes = np.load("classes.npy", allow_pickle=True)
    return model, classes

model, classes = load_model_and_classes()

# 2. Upload File
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("Analyze Sound"):
        with st.spinner("Analyzing audio..."):
            try:
                # Load audio
                y, sr = librosa.load(uploaded_file, sr=22050)
                
                # Fix audio signal length so spectrogram ALWAYS generates 35840 values
                # 128 mel bins * 280 time frames = 35840 features
                target_samples = 280 * 512  # hop_length is 512 by default
                y_fixed = librosa.util.fix_length(y, size=target_samples)
                
                # Extract Mel Spectrogram
                mel_spec = librosa.feature.melspectrogram(y=y_fixed, sr=sr, n_mels=128)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Reshape to exact 1D vector required by model: (1, 35840)
                X = mel_spec_db.flatten().reshape(1, 35840)
                
                # Model Prediction
                predictions = model.predict(X)[0]
                top_idx = np.argsort(predictions)[::-1][:5]
                
                st.subheader("Prediction Results:")
                for idx in top_idx:
                    st.write(f"**{classes[idx]}**: {predictions[idx]*100:.2f}%")
            
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
