import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

st.title("🤖 Face & Hand Landmark Detection Web App")

# Sidebar instructions
st.sidebar.header("📌 Instructions")
st.sidebar.write("""
- Choose to use your webcam or upload an image/video.
- The app will detect face mesh and hand landmarks.
- Landmarks will be drawn on the image/video.
""")

# MediaPipe Holistic setup
mp_holistic = mp.solutions.holistic
st.write("MediaPipe:", mp)
st.write("Version:", getattr(mp, "__version__", "Unknown"))
st.write("File:", getattr(mp, "__file__", "Unknown"))
st.write("Has solutions:", hasattr(mp, "solutions"))
st.write(dir(mp))
mp_drawing = mp.solutions.drawing_utils

holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Options: Webcam or upload
option = st.radio("Choose input source:", ["Webcam", "Upload Image/Video"])

if option == "Webcam":
    stframe = st.empty()
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            st.warning("Failed to access webcam.")
            break

        # Resize and convert to RGB
        frame = cv2.resize(frame, (800, 600))
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Process with MediaPipe
        results = holistic_model.process(image_rgb)
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr, results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(255,0,255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0,255,255), thickness=1)
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Show FPS
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        cv2.putText(image_bgr, f"{fps} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Display in Streamlit
        stframe.image(image_bgr, channels="BGR")

    capture.release()

elif option == "Upload Image/Video":
    uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        # Check if image or video
        if uploaded_file.name.endswith((".jpg", ".jpeg", ".png")):
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic_model.process(image_rgb)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Draw landmarks
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr, results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(255,0,255), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0,255,255), thickness=1)
                )
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            st.image(image_bgr, channels="BGR", caption="Detected Landmarks")

        else:  # Video
            st.warning("Video upload support is coming soon!")

st.markdown("---")
st.markdown("👨‍💻 *Created by Naps | Powered by OpenCV, Streamlit & MediaPipe*")
