
# Emotion Detection from Audio using Deep Learning

## Overview

This project focuses on recognizing human emotions from speech audio using deep learning model. In this project a Convolutional Neural Network (CNN) is trained on extracted audio features to classify emotions into multiple categories such as neutral, happy, sad, angry, calm, fear, disgust, and surprised.

---

## Project Structure

* Dataset Preprocessing(Labeling the data)
* Then Feature Extraction
* Printing top-5 dataframes
* Counted Values of different emotion labelled
* Shown the waveplot and spectrogram of different label
* Then used CNN model
* Ran some epochs to see the accuracy of particular epoch and val_accuracy
* Then generated the confusion matrix plot with f1 score and accuracy
* Shown accuracy of particular emotion prediction
* Generated test_model file and app.py file for streamlit

##  Workflow

### 1. Pre-processing

* Audio files are sampled using `librosa`
* 40 Mel-frequency cepstral coefficients (MFCCs) are extracted
* MFCCs are averaged across time to form a fixed-size feature vector

### 2. Model Pipeline

* **Model Type**: Convolutional Neural Network (CNN)
* **Input Shape**: (40, 1)
* **Layers**:

  * Conv1D + BatchNormalization + MaxPooling + Dropout (x2)
  * Flatten + Dense(128) + Dropout
  * Output Dense(6) with Softmax activation
* **Loss Function**: `categorical_crossentropy`
* **Optimizer**: `Adam`
* **Accuracy Achieved**: \~90% on validation set

---

## Testing the Model

Run the following script to predict emotion from a .wav file:

```bash
python test_model.py
```

Make sure your test .wav file is short (1-3 sec), 16kHz mono, and is placed in the same directory or path is updated in `test_model.py`.

---

##  Confusion Matrix & Accuracy Metrics

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 90%   |
| Precision | 88%   |
| Recall    | 87%   |
| F1-Score  | 87%   |


---

## Web App (Streamlit)

To run the web app:

* First make a folder including the trained model(emotion_cnn_model.h5) and app.py file.
* Then open terminal in that folder and type: streamlit run app.py
* The streamlit site will open in browser and you can browse the dataset to get result.

Features:

* Upload `.wav` file
* Audio is processed and fed into the model
* Returns predicted emotion in real-time

---

## Deployment Demo

A 2-minute demo video is included in the submission, showcasing:

* How to use the test script
* Using the Streamlit web app for prediction

---

## Dependencies

* TensorFlow / Keras
* NumPy
* Librosa
* Streamlit
* scikit-learn (for evaluation)

## 🎓 Author

**Arvind Verma**


---

