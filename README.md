# Face Mask Detection with Deep Learning 🧠😷

A real-time face mask detection system built with **Python**, **OpenCV**, and **TensorFlow/Keras**.  
This project detects whether a person is wearing a mask through webcam video stream using a trained MobileNetV2 model.

![demo](https://img.shields.io/badge/status-working-success) ![python](https://img.shields.io/badge/python-3.8-blue)

## 🔍 Features
- Detects faces and classifies mask vs. no-mask in real time.
- Utilizes **MobileNetV2** as the base model for transfer learning.
- Implements OpenCV for image processing and visualization.
- Trained on a custom dataset of masked and unmasked face images.

## 📹 Demo Video
👉 [Watch demo on YouTube](https://www.youtube.com/watch?v=-b68jGFwHsg)

## 🛠 Technologies Used
- Python 3.8  
- TensorFlow / Keras  
- OpenCV  
- MobileNetV2  
- NumPy, Matplotlib, scikit-learn

## 📁 Project Structure
.
├── dataset/
│   ├── with_mask/
│   └── without_mask/
├── face_detector/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── train_mask_detect.py
├── realtime_mask_detect.py
├── plot.png
└── mask_detect.model

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face-mask-detection.git
   cd face-mask-detection
   pip install -r requirements.txt
   python train_mask_detect.py
   python realtime_mask_detect.py
   
📈 Training Result
Accuracy: ~99%
Loss: converges well with minimal overfitting.

👤 Authors
Trần Nguyễn Khánh Hoàng

Trần Văn Đạt

📄 License
This project is for educational purposes only.
---
