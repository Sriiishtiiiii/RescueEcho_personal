# RescueEcho  
**AI-Driven Micro-Doppler Radar System for Human Detection in Disaster Scenarios**

---

## 📌 Overview

**RescueEcho** is an advanced AI-powered target classification system designed to assist search-and-rescue teams in locating survivors trapped under debris or operating in low-visibility environments such as collapsed buildings, smoke-filled areas, or nighttime disaster zones.

The system leverages **micro-Doppler radar signatures** to detect and analyze motion patterns that are characteristic of human activity. By combining **radar signal processing**, **spectrogram-based feature extraction**, and **machine learning classification**, RescueEcho can reliably distinguish **humans from non-living objects** and further classify **types of human motion** such as walking, limping, and crawling.

Unlike vision-based systems, RescueEcho is **non-line-of-sight**, lighting-independent, and robust to dust, smoke, and occlusion—making it highly suitable for disaster response scenarios.

---

## 🎯 Problem Statement

Traditional survivor detection techniques face significant limitations:
- Optical cameras fail in darkness, smoke, or debris.
- Thermal imaging struggles with heat diffusion and false positives.
- Manual search is time-consuming and dangerous for rescuers.

**RescueEcho addresses these challenges** by using radar-based sensing combined with AI to:
- Detect minute human movements under rubble
- Reduce false alarms from non-living objects
- Provide actionable intelligence in real time

---

## 🚀 Key Features

### 1. Micro-Doppler Signature Analysis
- Exploits micro-Doppler effects caused by subtle human movements (limbs, breathing, posture shifts)
- Differentiates periodic and non-periodic motion patterns
- Enables discrimination between biological and mechanical motion

### 2. Spectrogram-Based Learning
- Raw radar signals are transformed into **time–frequency spectrograms**
- Spectrograms capture motion dynamics that are well-suited for CNN-based learning
- Class-wise dataset organization (walking, limping, crawling, non-human)

### 3. Machine Learning–Driven Classification
- CNN-based deep learning model trained on spectrogram images
- Learns spatial–temporal patterns in frequency shifts
- Scalable to additional motion classes or environments

### 4. Real-Time Detection and Alerts
- Continuous radar signal acquisition
- On-the-fly preprocessing and inference
- Immediate detection feedback for rescue teams

### 5. Hardware–Software Co-Design
- Radar-based sensing integrated with embedded control
- Modular design allows easy upgrades or sensor replacement
- Edge-compatible architecture for field deployment

---

## 🛠️ Technology Stack

### Hardware
- **Infineon BGT24LTR11 Radar Sensor**
  - 24 GHz FMCW radar
  - High sensitivity to micro-motions
- **Arduino Uno**
  - Sensor interfacing and control
  - Data transmission and synchronization

### Software
- **Programming Languages**
  - Python (signal processing, ML, inference)
  - C++ (embedded control, low-level operations)

- **Libraries & Frameworks**
  - NumPy, Pandas – numerical computation and data handling
  - OpenCV – image preprocessing and transformations
  - TensorFlow / PyTorch – deep learning model training and evaluation
  - SciPy – FFT, STFT, and signal analysis

---

## 📊 Data Processing Pipeline

### 1. Radar Signal Acquisition
- Continuous-wave radar captures reflected signals
- Doppler shifts introduced by moving targets
- Micro-Doppler components encode fine-grained motion

### 2. Preprocessing
- Noise filtering and signal conditioning
- Segmentation into time windows
- Short-Time Fourier Transform (STFT)

### 3. Spectrogram Generation
- Frequency vs time representation
- Log-scaling and normalization
- Converted into image format for CNN input

### 4. Feature Learning
- CNN automatically extracts discriminative patterns
- Captures motion periodicity, spread, and intensity
- Avoids manual feature engineering

### 5. Classification & Decision Logic
- Outputs probability scores per class
- Threshold-based human detection
- Triggers alert when human presence is detected

---

Each folder contains spectrogram images derived from radar signals corresponding to the motion class.

---

## ⚙️ Model Overview

- Input: Spectrogram images (time–frequency domain)
- Architecture: Convolutional Neural Network
- Loss Function: Categorical Cross-Entropy
- Optimizer: Adam
- Evaluation Metrics:
  - Accuracy
  - Precision / Recall
  - Confusion Matrix

The model is designed to be lightweight and deployable on edge or near-edge systems.

---

## 🤝 Use Cases

- Earthquake and building collapse rescue
- Fire and smoke-filled environments
- Underground or tunnel rescues
- Military and defense reconnaissance
- Smart surveillance and safety systems

---

## 🙌 Acknowledgements

RescueEcho integrates concepts from:
- Radar signal processing
- Micro-Doppler phenomenology
- Computer vision and deep learning
- Embedded systems and hardware–software co-design

---



