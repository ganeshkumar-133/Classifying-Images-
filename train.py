import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

# --- CONFIGURATION ---
DATASET_DIR = 'dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_SAVE_PATH = 'saved_model'

# 1. PREPARE DATA
# We use Data Augmentation to artificially increase dataset size and prevent overfitting
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, # 80% train, 20% validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

print("Loading Training Data...")
train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse', # Use sparse for SVM labels
    subset='training'
)

print("Loading Validation Data...")
val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    shuffle=False # Important for evaluation later
)

# 2. BUILD CNN FEATURE EXTRACTOR
# We use MobileNetV2 pre-trained on ImageNet for high accuracy
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
feature_extractor = Model(inputs=base_model.input, outputs=x)

# 3. EXTRACT FEATURES
print("Extracting features from Training data (this may take a moment)...")
X_train, y_train = [], []
for images, labels in train_generator:
    features = feature_extractor.predict(images, verbose=0)
    X_train.append(features)
    y_train.append(labels)
    if len(X_train) * BATCH_SIZE >= train_generator.samples:
        break

X_train = np.vstack(X_train)
y_train = np.hstack(y_train)

print("Extracting features from Validation data...")
X_val, y_val = [], []
for images, labels in val_generator:
    features = feature_extractor.predict(images, verbose=0)
    X_val.append(features)
    y_val.append(labels)
    if len(X_val) * BATCH_SIZE >= val_generator.samples:
        break

X_val = np.vstack(X_val)
y_val = np.hstack(y_val)

# 4. TRAIN SVM CLASSIFIER
print("Training SVM Classifier...")
# C is regularization parameter. Higher C = strict fit.
svm_model = SVC(kernel='linear', C=1.0, probability=True)
svm_model.fit(X_train, y_train)

# 5. EVALUATE
print("Evaluating Model...")
y_pred = svm_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
class_labels = list(range(len(train_generator.class_indices)))
print("\nClassification Report:\n", classification_report(
    y_val,
    y_pred,
    labels=class_labels,
    target_names=list(train_generator.class_indices.keys()),
    zero_division=0,
))

# 6. SAVE MODEL
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

# Save the CNN (for feature extraction)
feature_extractor.save(os.path.join(MODEL_SAVE_PATH, 'cnn_feature_extractor.h5'))
# Save the SVM (for classification)
joblib.dump(svm_model, os.path.join(MODEL_SAVE_PATH, 'svm_classifier.pkl'))
# Save class indices mapping
joblib.dump(train_generator.class_indices, os.path.join(MODEL_SAVE_PATH, 'class_indices.pkl'))

print("Model saved successfully in 'saved_model' folder.")