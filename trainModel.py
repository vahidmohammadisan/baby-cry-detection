import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping

# Function to extract MFCC features from audio files
def extract_mfcc(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    audio, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfccs.T

# Function to prepare data and labels for binary classification (crying vs. silence)
def prepare_binary_data(data_dir_crying, data_dir_silence, label_crying=1, label_silence=0):
    X, y = [], []
    
    # Load crying data
    for filename in os.listdir(data_dir_crying):
        if filename.endswith(".wav"):
            file_path = os.path.join(data_dir_crying, filename)
            mfccs = extract_mfcc(file_path)
            X.append(mfccs)  # Append the 2D MFCC data
            y.append(label_crying)
    
    # Load silence data
    for filename in os.listdir(data_dir_silence):
        if filename.endswith(".wav"):
            file_path = os.path.join(data_dir_silence, filename)
            mfccs = extract_mfcc(file_path)
            X.append(mfccs)  # Append the 2D MFCC data
            y.append(label_silence)
    
    return X, y

# Define paths to your crying and silence audio data
crying_dir = "/home/vahid/projects/projects/cry_detection/crying"
silence_dir = "/home/vahid/projects/projects/cry_detection/silence"

# Prepare data and labels for crying and silence classes
X_data, y_data = prepare_binary_data(crying_dir, silence_dir, label_crying=1, label_silence=0)

# Check if data is not empty
if len(X_data) == 0:
    raise ValueError("No audio data found. Please check your data paths.")

# Split the combined data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Convert labels to NumPy arrays
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

# Reshape input data to match the expected format for 1D CNN
X_train = np.array(X_train).reshape(-1, X_train[0].shape[0], X_train[0].shape[1])
X_val = np.array(X_val).reshape(-1, X_val[0].shape[0], X_val[0].shape[1])
X_test = np.array(X_test).reshape(-1, X_test[0].shape[0], X_test[0].shape[1])

# Build a simple CNN model for binary classification (crying vs. silence)
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1], X_train.shape[2])),  # Adjust input shape
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(128, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(256, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train, y_train, epochs=5000, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")

# Save the trained model for later use
model.save("crying_vs_silence_model.h5")
