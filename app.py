import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
from scipy.ndimage import zoom

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
                # Target audio duration: Exact 5 Seconds at 22050 Hz
                target_sr = 22050
                target_len = target_sr * 5
                
                y, sr = librosa.load(uploaded_file, sr=target_sr)
                
                # Pad or truncate audio signal to exact 5 seconds
                if len(y) < target_len:
                    y = np.pad(y, (0, target_len - len(y)))
                else:
                    y = y[:target_len]
                
                # Extract Mel Spectrogram
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Try passing 2D Spectrogram directly or dynamically reshaped vector
                try:
                    # Target shape 1: Exact target frame size (128, 170) -> Conv layers shrink to 35840
                    target_height, target_width = 128, 170
                    zoom_h = target_height / mel_spec_db.shape[0]
                    zoom_w = target_width / mel_spec_db.shape[1]
                    spec_resized = zoom(mel_spec_db, (zoom_h, zoom_w))
                    
                    X = np.expand_dims(spec_resized, axis=-1)
                    X = np.expand_dims(X, axis=0)
                    predictions = model.predict(X)[0]
                except Exception:
                    # Target shape 2: Flattened direct feature vector
                    X_flat = mel_spec_db.flatten()[:35840].reshape(1, 35840)
                    predictions = model.predict(X_flat)[0]
                
                top_idx = np.argsort(predictions)[::-1][:5]
                
                st.subheader("Prediction Results:")
                for idx in top_idx:
                    st.write(f"**{classes[idx]}**: {predictions[idx]*100:.2f}%")
            
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
