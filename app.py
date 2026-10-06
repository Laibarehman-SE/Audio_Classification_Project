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
                # Load Audio at 22050 Hz
                y, sr = librosa.load(uploaded_file, sr=22050)
                
                # Force audio sample count to exactly match 280 Mel spectrogram frames
                # (280 frames * 512 hop_length = 143,360 samples)
                target_samples = 280 * 512
                y_fixed = librosa.util.fix_length(y, size=target_samples)
                
                # Generate Mel Spectrogram (128 Mel Bins x 280 Frames = 35,840 Total Elements)
                mel_spec = librosa.feature.melspectrogram(y=y_fixed, sr=sr, n_mels=128, hop_length=512)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Make sure shape is exactly (128, 280)
                mel_spec_db = mel_spec_db[:, :280]
                
                # Attempt 1: Flattened 1D Input Shape (1, 35840)
                try:
                    X = mel_spec_db.flatten().reshape(1, 35840)
                    predictions = model.predict(X)[0]
                except Exception:
                    # Attempt 2: 4D Conv Input Shape (1, 128, 280, 1)
                    X = np.expand_dims(mel_spec_db, axis=-1)
                    X = np.expand_dims(X, axis=0)
                    predictions = model.predict(X)[0]
                
                # Output Top 5 Predictions
                top_idx = np.argsort(predictions)[::-1][:5]
                
                st.subheader("Prediction Results:")
                for idx in top_idx:
                    st.write(f"**{classes[idx]}**: {predictions[idx]*100:.2f}%")
            
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
