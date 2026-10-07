import gradio as gr
import numpy as np
import tensorflow as tf
import librosa

# 1. Exact Model & Classes Load
MODEL_PATH = "audio_classification_cnn.h5"
CLASSES_PATH = "classes.npy"

model = tf.keras.models.load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)

# 2. Prediction Function (Fixed Shape for CNN)
def predict_audio(audio_path):
    if audio_path is None:
        return None

    # Audio Load & Resample (22.05 kHz)
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # Silence Trimming
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)

    # Mel-Spectrogram Extraction
    mel_spec = librosa.feature.melspectrogram(y=y_trimmed, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Padding / Truncating to exact width (174)
    target_width = 174
    current_width = mel_spec_db.shape[1]

    if current_width < target_width:
        pad_width = target_width - current_width
        mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :target_width]

    # Shape matching CNN Model Input: (1, 128, 174, 1)
    spec_input = np.expand_dims(mel_spec_db, axis=-1)  # Channel axis
    spec_input = np.expand_dims(spec_input, axis=0)    # Batch axis

    # Prediction
    preds = model.predict(spec_input)[0]

    # Top 5 Predictions Extraction
    top_5_idx = np.argsort(preds)[-5:][::-1]
    
    results = {str(classes[i]).replace('_', ' ').title(): float(preds[i]) for i in top_5_idx}
    return results

# 3. UI Styling & Layout
custom_css = """
body, .gradio-container {
    background-color: #12181f !important;
    color: #e2e8f0 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.main-header {
    text-align: center;
    margin-bottom: 24px;
    padding-top: 10px;
}

.main-header .title-wrapper {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    font-size: 2.2rem;
    font-weight: 700;
}

.main-header .brand-text { color: #ffffff !important; }
.main-header .ai-tag { color: #64748b !important; font-weight: 500; }
.main-header p { color: #94a3b8 !important; font-size: 0.95rem; margin-top: 4px; }

.card-panel {
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    background: #1a202c !important;
}

.card-title {
    color: #ffffff !important;
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 12px;
}

.stat-item .bar { background-color: #10b981 !important; }
button.primary { background-color: #10b981 !important; border: none !important; }
"""

custom_theme = gr.themes.Soft(primary_hue="emerald", neutral_hue="slate")

with gr.Blocks(theme=custom_theme, css=custom_css) as interface:
    gr.Markdown(
        """
        <div class="main-header">
            <div class="title-wrapper">
                <span>🎙️</span>
                <span class="brand-text">AcoustiSense</span>
                <span class="ai-tag">AI</span>
            </div>
            <p>Environmental Sound Classification System</p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(elem_classes=["card-panel"]):
            gr.Markdown("<div class='card-title'>Input & Visualization</div>")
            file_input = gr.File(label="DROP AUDIO FILE OR CLICK TO UPLOAD", file_count="single", type="filepath")
            submit_btn = gr.Button("Analyze Sound", variant="primary")
            audio_display = gr.Audio(
                label="Audio Visualization", 
                interactive=False,
                waveform_options={"waveform_color": "#10b981", "waveform_progress_color": "#059669"}
            )
            
        with gr.Column(elem_classes=["card-panel"]):
            gr.Markdown("<div class='card-title'>Prediction Dashboard</div>")
            gr.Markdown("Top 5 Classes:")
            label_output = gr.Label(num_top_classes=5, label="")

    file_input.change(fn=lambda x: x, inputs=file_input, outputs=audio_display)
    submit_btn.click(fn=predict_audio, inputs=file_input, outputs=label_output)

import streamlit as st
import streamlit.components.v1 as components

# Interface launch
interface.launch(prevent_thread_lock=True, server_name="0.0.0.0", server_port=7860)

# Streamlit Page Display
st.set_page_config(page_title="AcoustiSense AI", layout="wide")
components.iframe("http://localhost:7860", height=800, scrolling=True)
