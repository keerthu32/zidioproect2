import cv2
import numpy as np
from keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
import mediapipe as mp
import librosa
import pyaudio
import wave
import os
import streamlit as st
from textblob import TextBlob

# Load pre-trained facial emotion detection model
emotion_model = load_model(r"E:\my_model.keras")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Function for text sentiment analysis
def analyze_text_sentiment(input_text):
    sentiment = TextBlob(input_text).sentiment
    if sentiment.polarity > 0:
        return "Positive"
    elif sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Function for live camera video emotion detection
def detect_emotion_from_video():
    cap = cv2.VideoCapture(0)  # Open webcam
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to grab frame")
                break

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            # Process face detection and predict emotions
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)

                    # Extract face for emotion prediction
                    face = frame[y:y+height, x:x+width]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face = cv2.resize(face, (48, 48))
                    face = face.astype("float") / 255.0
                    face = img_to_array(face)
                    face = np.expand_dims(face, axis=0)

                    # Predict emotion
                    preds = emotion_model.predict(face)[0]
                    emotion = emotion_labels[np.argmax(preds)]

                    # Draw the bounding box and emotion label
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Display the frame in Streamlit
            stframe.image(frame, channels="BGR")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

# Function to record audio for speech emotion detection
def record_audio(output_file="output.wav", duration=5):
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

    st.write("Recording...")
    frames = []
    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    st.write("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save audio
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Function for speech emotion detection
def detect_emotion_from_speech(audio_file="output.wav"):
    # Load audio file
    y, sr = librosa.load(audio_file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    st.write("MFCC Features extracted for emotion detection.")
    # Implement ML or DL model here for audio emotion classification
    # Dummy prediction for illustration
    st.write("Predicted Speech Emotion: Neutral")

# Streamlit App Interface
st.title("Emotion Detection Application")

mode = st.selectbox("Choose mode:", ["Video Emotion Detection", "Speech Emotion Detection", "Text Sentiment Analysis"])

if mode == "Video Emotion Detection":
    st.write("Press 'q' in the video window to quit.")
    detect_emotion_from_video()
elif mode == "Speech Emotion Detection":
    duration = st.slider("Select recording duration (seconds):", 1, 10, 5)
    if st.button("Record and Detect Emotion"):
        record_audio(duration=duration)
        detect_emotion_from_speech()
elif mode == "Text Sentiment Analysis":
    input_text = st.text_input("Enter the text for sentiment analysis:")
    if input_text:
        sentiment = analyze_text_sentiment(input_text)
        st.write(f"Predicted Text Sentiment: {sentiment}")
