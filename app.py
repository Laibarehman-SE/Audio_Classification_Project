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
                # Process audio signal
                y, sr = librosa.load(uploaded_file, duration=3.0, sr=22050)
                
                # Fixed audio length padding/cropping (3 seconds = 66150 samples)
                target_length = 22050 * 3
                if len(y) < target_length:
                    y = np.pad(y, (0, target_length - len(y)))
                else:
                    y = y[:target_length]
                
                # Mel-Spectrogram extraction
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Check expected input shape of model
                input_shape = model.input_shape
                
                # Reshape tensor
                if len(input_shape) == 4:
                    # Model expects (batch, height, width, channels)
                    X = np.expand_dims(mel_spec_db, axis=-1)
                    X = np.expand_dims(X, axis=0)
                else:
                    X = np.expand_dims(mel_spec_db, axis=0)
                
                # Prediction
                predictions = model.predict(X)[0]
                top_idx = np.argsort(predictions)[::-1][:5]
                
                st.subheader("Prediction Results:")
                for idx in top_idx:
                    st.write(f"**{classes[idx]}**: {predictions[idx]*100:.2f}%")
            
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
