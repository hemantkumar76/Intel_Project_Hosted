import streamlit as st
import cv2
import torch
import numpy as np
import tempfile
import os
from ultralytics import YOLO


def get_model_path(model_name):
    return os.path.join(os.path.dirname(__file__), model_name)


model1_path = get_model_path('models/best_1.pt')
model2_path = get_model_path('models/best_2.pt')
model1 = YOLO(model1_path)
model2 = YOLO(model2_path)

st.title("Innovative Monitoring System for TeleICU Patients Using Video Processing and Deep Learning Made By Hemant")

st.write("""
         This application allows you to upload a video for:
         1. Detecting people in an ICU Ward (Doctor, Family Member, Nurse, Patient) using YOLO model.
         2. Detecting Patient movement when He/She is Alone.
         """)

option = st.selectbox("Select an option", ("Detect People", "Detect Movement"))

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

# people detect karne wlaa function for the statement 1 
def detect_people(video_path, model):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    out = None
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'H264'), fps, (frame_width, frame_height))


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        boxes = results[0].boxes.data.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            predictions.append({'box': [x1, y1, x2, y2], 'confidence': conf, 'class': model.names[int(cls)]})
            label = f"{model.names[int(cls)]}: {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        out.write(frame)

    cap.release()
    out.release()
    return predictions, out_path
# moment detection ka function xyxy function hai or may be lambda function hai
def detect_movement(video_path, threshold=30):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    movement_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is None:
            prev_frame = gray_frame
            continue

        frame_diff = cv2.absdiff(prev_frame, gray_frame)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        if np.sum(thresh) > 0:  
            movement_detected = True
            break

        prev_frame = gray_frame

    cap.release()
    return movement_detected

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    st.video(tfile.name)

    st.write("Processing video Hold on...")
    
    if option == "Detect People":
        results1, annotated_video_path = detect_people(tfile.name, model1)
        
        st.write("Model 1 (People Detection) Results:")
        for result in results1:
            st.write(f"{result['class']} - Confidence: {result['confidence']}")
        
        st.write("Annotated video with bounding boxes:")
        st.video(annotated_video_path)
    
    elif option == "Detect Movement":
        results2 = detect_movement(tfile.name)
        
        st.write("Model 2 (Movement Detection) Result:")
        st.write("No movement detected" if results2 else "Movement detected")

        
st.markdown("Made with :heart: by Hemant")
