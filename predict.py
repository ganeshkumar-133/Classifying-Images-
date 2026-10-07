import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import joblib
import os

# --- CONFIGURATION ---
MODEL_DIR = 'saved_model'
IMG_SIZE = (224, 224)

def load_predictor():
    cnn_model = load_model(os.path.join(MODEL_DIR, 'cnn_feature_extractor.h5'))
    svm_model = joblib.load(os.path.join(MODEL_DIR, 'svm_classifier.pkl'))
    class_indices = joblib.load(os.path.join(MODEL_DIR, 'class_indices.pkl'))
    labels_map = {v: k for k, v in class_indices.items()}
    return cnn_model, svm_model, labels_map

def classify_image(img_source, cnn_model, svm_model, labels_map):
    img = image.load_img(img_source, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    features = cnn_model.predict(img_array, verbose=0)
    prediction = svm_model.predict(features)[0]
    probs = svm_model.predict_proba(features)[0]
    predicted_class = labels_map[prediction]
    probability_by_class = {
        labels_map[class_index]: float(probability)
        for class_index, probability in zip(svm_model.classes_, probs)
    }
    return {
        'category': predicted_class,
        'confidence': float(np.max(probs)),
        'probabilities': probability_by_class,
    }

def predict_image(img_path):
    print("Loading models...")
    result = classify_image(img_path, *load_predictor())
    print("-" * 30)
    print(f"Image: {img_path}")
    print(f"Prediction: {result['category'].upper()}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print("-" * 30)
    return result['category']

# --- TESTING ---
# Change 'test_image.jpg' to the path of the image you want to test
if __name__ == "__main__":
    # Example usage:
    # predict_image('path/to/your/test_shoe.jpg')
    test_path = input("Enter the path to the image file: ")
    if os.path.exists(test_path):
        predict_image(test_path)
    else:
        print("File not found.")