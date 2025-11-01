# 🧠 Study of Deep Learning Models for the Classification of Epileptic Patients

### 📍 End of Year Project 2 — RISC Laboratory, ENIT

**Duration:** October 2024 – April 2025

---

## 🩺 Overview

This project presents a comparative study of deep learning architectures for classifying **epileptic patients** based on **EEG data**.
The goal is to explore and evaluate different preprocessing and modeling strategies to enhance the accuracy and clinical relevance of EEG-based epilepsy detection.
A simple **Streamlit interface** is provided to classify EEG data in real time, making the system suitable for practical use in clinical decision support.

---

## 🧩 Objectives

* Analyze EEG data and extract relevant features for classification.
* Compare the performance of multiple deep learning architectures.
* Evaluate the impact of different preprocessing and optimization techniques.
* Develop a user-friendly interface for real-time EEG classification.

---

## 🧠 Deep Learning Architectures Implemented

1. **Artificial Neural Network (ANN)**
2. **Convolutional Neural Network (CNN)**
3. **Hybrid CNN-LSTM Model**

Each model was trained and tested under various conditions to compare classification accuracy, precision, recall, and computational efficiency.

---

## ⚙️ Preprocessing Techniques

* **Wavelet Decomposition:** for time-frequency analysis of EEG signals.
* **Third-Order Derivative Filtering:** to enhance subtle signal variations.

These methods improved the signal-to-noise ratio and supported more robust feature extraction.

---

## 📊 Evaluation Metrics

* **Accuracy**
* **Precision**
* **Recall**
* **Training Time**

Experiments included **binary** and **multi-class** classifications across six dataset configurations.

---

## 💻 Technologies Used

* **Programming Language:** Python
* **Frameworks & Libraries:** TensorFlow, Keras, Scikit-learn
* **Data Processing:** NumPy, PyWavelet
* **Visualization:** Matplotlib
* **Interface:** Streamlit
* **Development Environment:** Jupyter Notebook

---

## 🧪 Results Summary

The **CNN-LSTM hybrid model** achieved the most consistent performance across preprocessing methods, showing improved generalization and faster convergence.
Wavelet preprocessing was particularly effective for reducing noise and improving feature discriminability in EEG signals.

---

## 🚀 How to Run the Interface

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/Study-of-Deep-Learning-Models-for-the-Classification-of-Epileptic-Patients.git
   ```
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

## 🧑‍🔬 Authors

**Ayoub Abidi**
Master’s Student – National Engineering School of Tunis (ENIT)
RISC Laboratory

---

## 📚 Keywords

`EEG` · `Deep Learning` · `Epilepsy` · `CNN` · `LSTM` · `Signal Processing` · `Wavelet` · `Biomedical AI`

---
