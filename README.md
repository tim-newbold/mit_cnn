# 😃 MIT Advanced Data Science Capstone – Facial Emotion Recognition (CNN)

This project showcases a deep learning pipeline for **facial emotion recognition** using a custom Convolutional Neural Network (CNN). Built as part of the **MIT Advanced Data Science Program**, the model is capable of accurately identifying human emotions in facial imagery—laying the foundation for real-time applications in areas like mental health, customer experience, and security systems.

> 🧠 **Goal**: Train a robust CNN to classify facial emotions with high accuracy  
> 🧰 **Built With**: TensorFlow · OpenCV · Flask · Python · Keras  
> 📈 **Focus Areas**: Model tuning · Real-time inference · Human-centered AI

---

## 🧩 Project Overview

This capstone combines advanced CNN modeling, thoughtful optimization techniques, and real-time deployment potential. The final model effectively classifies facial expressions into core emotions (e.g., happy, sad, surprised, neutral) from grayscale input images.

### 🌟 Key Features

- **🔬 Model Architecture**  
  - Five-layer CNN designed for feature-rich extraction  
  - Uses **Global Average Pooling**, **Batch Normalization**, and **Dropout** for better generalization and regularization

- **⚙️ Training Enhancements**  
  - **Focal Loss** to handle class imbalance and focus on hard-to-classify images  
  - **Dynamic learning rates** and **early stopping** to prevent overfitting  
  - Trained on clean, labeled emotion datasets with cross-validation

- **🚀 Real-Time Ready**  
  - Integrated with **OpenCV** for live face detection  
  - Delivered via a lightweight **Flask API** for real-time emotion prediction in production environments

---

## 🧱 Model Architecture Overview

```text
Input: Grayscale facial image (e.g., 48x48 pixels)

[Conv2D] → [BatchNorm] → [ReLU]  
→ [Conv2D] → [BatchNorm] → [MaxPooling]  
→ [Conv2D] → [Dropout] → [GlobalAvgPooling]  
→ [Dense Layer]  
→ [Softmax Layer] (Emotion Probabilities)
