import pyaudio
from tensorflow import keras
import librosa
import numpy as np
import serial

# Function to extract MFCC features from audio frames
def extract_mfcc(audio_frame, sr, n_mfcc=13, n_fft=2048, hop_length=512):
    mfccs = librosa.feature.mfcc(y=audio_frame, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfccs.T

# Load the trained model
model = keras.models.load_model("crying_vs_silence_model.h5")  # Use the updated model file

# Set PyAudio parameters
format = pyaudio.paInt16
channels = 1
rate = 44100  # Set the sample rate to match your training data
chunk_size = int(rate * 5)  # Ensure chunk size is an integer

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk_size)

# Define class labels for binary classification (crying vs. silence)
class_labels = ["silence , noise", "is crying..."]  # Updated class labels

# Initialize serial communication with Arduino
arduino_port = "/dev/ttyACM0"  # Change this to your Arduino port
arduino_baudrate = 9600
arduino = serial.Serial(arduino_port, arduino_baudrate, timeout=1)

try:
    print("Streaming...")

    while True:
        # Read audio data from the microphone
        audio_data = stream.read(chunk_size)

        # Convert the audio data to floating-point format
        audio_array = np.frombuffer(audio_data, dtype=np.int16) / 32767.0

        # Extract MFCC features from the audio frame
        mfccs = extract_mfcc(audio_array, rate)

        # Reshape the data to match the input shape of the model
        mfccs = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1])

        # Make predictions on the audio frame
        predictions = model.predict(mfccs)

        # Get the predicted class label
        predicted_class_label = class_labels[int(predictions[0][0] > 0.5)]  # Assuming 0.5 as the threshold for binary classification

        print(f"Predicted class: {predicted_class_label}, Probability: {predictions[0][0]}")

        # Send the prediction to Arduino
        if predicted_class_label == "is crying...":
            arduino.write(b'1')  # Send 1 to Arduino
        else:
            arduino.write(b'0')  # Send 0 to Arduino

except KeyboardInterrupt:
    print("Streaming stopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    arduino.close()
