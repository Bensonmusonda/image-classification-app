# 📱 Image Classification Mobile App

An AI-powered image classification app built with **Flutter** and **TensorFlow Lite**. It identifies captured images as one of four objects: **controller**, **leaf**, **mouse**, or **pen**. The core model was trained using **transfer learning** on **MobileNetV2**, and the full pipeline runs directly on-device for fast and private inference.

## 🚀 Features

- Live camera feed
- Real-time image capture and classification
- On-device TensorFlow Lite model (no server needed)
- Predicts object class and confidence level
- Easy retake and re-classify options
- Built for Android using Flutter

---

## 🧠 Model Development

**Platform:** Google Colab  
**Framework:** TensorFlow / Keras  
**Architecture:** MobileNetV2 + Custom Dense Head  
**Optimizer:** Adam  
**Loss:** Categorical Crossentropy  
**Augmentation:** Rotation, Shear, Zoom, Flip, Shifts  

### 🗂 Dataset

The dataset was organized into 4 subdirectories (controller, leaf, mouse, pen).  
Data augmentation was applied using `ImageDataGenerator` to improve robustness.

### 🔄 Training Process

- MobileNetV2 used as frozen base model (ImageNet weights)
- Custom classification head: `GlobalAvgPool2D → Dense (ReLU) → Dense (Softmax)`
- Split: 80% training / 20% validation
- Evaluation done in Colab on both test images and unseen uploads

### 📦 Export

- Trained model exported as `.tflite` using TensorFlow Lite Converter
- Model saved to Google Drive for deployment into the mobile app

---

## 📲 Mobile App Development

**Framework:** Flutter  
**AI Integration:** `tflite_flutter` plugin  
**Model:** `object_classifier_model.tflite`  

### 🛠 Core Workflow

1. Launches live camera feed
2. Captures an image
3. Resizes image to 224x224 and normalizes pixel values
4. Performs inference with TFLite model
5. Displays predicted class and confidence
6. User can retake or re-classify

---

## 🧪 Testing

Tested on physical Android device with real-world images.  
Most predictions were correct, with minor edge cases (e.g., transparent vs. opaque objects).

---

## 🔍 Lessons Learned

- Transfer learning significantly speeds up training and improves performance
- Real-time AI inference on-device is feasible with lightweight models like MobileNetV2
- Data diversity is critical for robust performance
- Integrating AI into mobile apps with Flutter and TensorFlow Lite is efficient and scalable

---

## 🏁 Future Improvements

- Increase dataset size and diversity for better generalization
- Add support for more object classes
- Implement performance logging (inference time, confidence history)
- Improve UX with animations or interactive feedback

---

## 👨‍💻 Author

**Benson Musonda**  
AI Developer & Computer Science Student  
202204757  
Email: *[Bensonmusonda12@gmail.com]*  
GitHub: [https://github.com/Bensonmusonda] 

---

## 🏷 Tags

`#AI` `#MachineLearning` `#Flutter` `#TensorFlowLite` `#MobileApp` `#ImageClassification`

