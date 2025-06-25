from keras.models import load_model
import numpy as np
import librosa

# Loading the trained model
model = load_model("emotion_model.h5")

# Function to extract features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled.reshape(1, 40, 1)

# Path to test file 
file_path = "test_audio.wav"  
features = extract_features(file_path)

# Prediction
prediction = model.predict(features)
emotion_classes = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear','disgust','surprised']
predicted_class = emotion_classes[np.argmax(pred)]
print("Predicted Emotion:", predicted_class)
