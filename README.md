# RescueEcho

**RescueEcho** is an AI-powered micro-Doppler-based target classification system that uses radar signals to differentiate between humans and non-living objects in disaster scenarios. It helps identify survivors under debris or in low-visibility conditions, enhancing the efficiency of rescue operations.

## 🚀 Features
- **Micro-Doppler Signature Analysis**: Uses radar data to detect human motion patterns.
- **Machine Learning Model**: Trained on spectrogram images categorized into movements like walking, limping, and crawling.
- **Real-Time Detection**: Identifies survivors using radar sensors.
- **Hardware Integration**: Utilizes the **Infineon BGT24LTR11 radar sensor** and **Arduino Uno**.

## 📌 System Architecture
1. **Data Collection**: Radar captures micro-Doppler signatures.
2. **Preprocessing**: Converts radar signals into spectrogram images.
3. **Model Training**: Uses ML techniques for motion classification.
4. **Detection & Alert**: Real-time analysis for rescue operations.

## 🛠️ Technology Stack
- **Hardware**: Infineon BGT24LTR11 Radar, Arduino Uno
- **Programming Languages**: Python, C++
- **Libraries & Frameworks**:
  - OpenCV (for image processing)
  - NumPy, Pandas (data handling)
  - TensorFlow/PyTorch (ML model training)
- **Embedded Systems**: Arduino for sensor control

## 📊 Data Processing Workflow
1. **Radar Signal Acquisition**
2. **Spectrogram Generation**
3. **Feature Extraction**
4. **Classification using ML**
5. **Decision & Alert Mechanism**
