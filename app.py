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
            # Process audio
            y, sr = librosa.load(uploaded_file, duration=3.0, sr=22050)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Reshape for CNN model
            X = mel_spec_db[..., np.newaxis]
            X = np.expand_dims(X, axis=0)
            
            # Predict
            predictions = model.predict(X)[0]
            top_idx = np.argsort(predictions)[::-1][:5]
            
            st.subheader("Prediction Results:")
            for idx in top_idx:
                st.write(f"**{classes[idx]}**: {predictions[idx]*100:.2f}%")
