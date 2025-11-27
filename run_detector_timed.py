import mediapipe as mp
import cv2
import pickle
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import time

# Load the pre-trained model
print("Loading pre-trained model...")
with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)
print("Model loaded successfully!\n")

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Set duration
duration = 30  # Run for 30 seconds
print(f"Starting detector for {duration} seconds...")
print("Try different facial expressions and body postures!")
print("A window will open showing your webcam with detections\n")
time.sleep(2)  # Give user time to read

cap = cv2.VideoCapture(0)
emotions = []
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print(f"BODY LANGUAGE DETECTOR - Running for {duration} seconds")
print("="*70)
print(f"{'Time':<12} {'Emotion':<15} {'Confidence':<12} {'Bar Chart':<30}")
print("="*70)

frame_count = 0
start_time = time.time()
detection_count = 0

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("\nWarning: Failed to grab frame from camera")
            break

        # Check if time is up
        elapsed = time.time() - start_time
        if elapsed > duration:
            print("\nTime's up!")
            break

        frame_count += 1

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))

        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

        # Export coordinates and make predictions
        try:
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Extract Face landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            # Concatenate rows
            row = pose_row + face_row

            # Make Detections
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            max_prob = round(body_language_prob[np.argmax(body_language_prob)], 3)

            emotions.append(str(body_language_class))
            detection_count += 1

            # Print to terminal every 10 frames
            if frame_count % 10 == 0:
                timestamp = datetime.now().strftime("%H:%M:%S")
                bar = "█" * int(max_prob * 30)
                remaining = int((duration - elapsed))
                print(f"{timestamp:<12} {body_language_class:<15} {max_prob:<12.1%} {bar:<30} [{remaining}s left]")

            # Draw on image
            coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                        , [640,480]).astype(int))

            cv2.rectangle(image,
                          (coords[0], coords[1]+5),
                          (coords[0]+len(body_language_class)*20, coords[1]-30),
                          (245, 117, 16), -1)
            cv2.putText(image, body_language_class, coords,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(image, (0,0), (300, 80), (245, 117, 16), -1)
            cv2.putText(image, 'CLASS', (95,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0], (90,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'PROB', (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(max_prob), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show time remaining
            cv2.putText(image, f'{remaining}s', (220,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            if frame_count % 30 == 0:
                print(f"[{int(elapsed)}s] Waiting for clear face/body detection...")

        cv2.imshow('Body Language Detector - Auto-closing', image)

        # Check for 'q' key press to quit early
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("\nManually stopped by user")
            break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print("DETECTION SUMMARY")
print("="*70)
print(f"Duration: {int(elapsed)} seconds")
print(f"Total frames with detections: {detection_count}")

if emotions:
    emotion_counts = dict((x, emotions.count(x)) for x in set(emotions))
    print(f"\nEmotion distribution:")
    print("=" * 70)
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(emotions)) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {emotion:<15} {count:>5} frames  ({percentage:>5.1f}%)  {bar}")

    print("\nMost detected emotion: " + max(emotion_counts, key=emotion_counts.get))
else:
    print("No emotions detected. Make sure your face and upper body are visible to the camera.")

print("="*70)
