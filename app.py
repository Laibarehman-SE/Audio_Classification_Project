import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import cv2

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
                # Load Audio
                y, sr = librosa.load(uploaded_file, sr=22050)
                
                # Mel Spectrogram (128 Mel Bins)
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Force exact shape (128, 280) so total features = 128 * 280 = 35840
                resized_spec = cv2.resize(mel_spec_db, (280, 128))
                
                # Reshape for CNN Input: (1, 128, 280, 1)
                X = np.expand_dims(resized_spec, axis=-1)
                X = np.expand_dims(X, axis=0)
                
                # Model Prediction
                predictions = model.predict(X)[0]
                top_idx = np.argsort(predictions)[::-1][:5]
                
                st.subheader("Prediction Results:")
                for idx in top_idx:
                    st.write(f"**{classes[idx]}**: {predictions[idx]*100:.2f}%")
            
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
