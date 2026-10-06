# 🎵 AcoustiSense AI - Audio Classification Web App

AcoustiSense AI is an end-to-end Machine Learning web application designed to classify environmental sounds. The application utilizes a Convolutional Neural Network (CNN) paired with a Librosa feature extraction pipeline, deployed seamlessly on Streamlit Cloud.

---

## 🚀 Key Features

* **Real-time Audio Processing:** Support for loading and playing audio files (`.wav`, `.mp3`, `.ogg`) directly within the web interface.
* **Mel-Spectrogram Feature Extraction:** Converts audio signals into 128 Mel-frequency bins with fixed time frames using the Librosa library.
* **Deep Learning Inference:** Pre-trained CNN model predicts and displays the top 5 most likely audio classes along with their confidence probabilities.
* **Cloud Deployed:** Hosted on Streamlit Cloud with Git LFS support for large model files.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Web Framework:** Streamlit
* **Deep Learning:** TensorFlow / Keras
* **Audio Processing:** Librosa
* **Data Processing:** NumPy
* **Version Control & LFS:** Git & Git Large File Storage (Git LFS)

---

## 📁 Repository Structure

```text
├── app.py                           # Main Streamlit application entry point
├── requirements.txt                 # Python dependencies
├── audio_classification_cnn.h5      # Pre-trained CNN Model (Tracked via Git LFS)
├── classes.npy                      # Target class labels numpy array
├── .gitattributes                   # Git LFS tracking configuration
└── README.md                        # Project documentation
