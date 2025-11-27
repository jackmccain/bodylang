import mediapipe as mp
import cv2
import pickle
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

# Load the pre-trained model
print("Loading pre-trained model...")
with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)
print("Model loaded successfully!\n")

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
emotions = []
warnings.filterwarnings('ignore')

print("="*60)
print("BODY LANGUAGE DETECTOR - REAL-TIME OUTPUT")
print("="*60)
print("The detector is now running!")
print("- A window should open showing your webcam feed")
print("- Detections will be printed below in real-time")
print("- Press 'Q' in the video window to quit\n")
print("="*60)
print(f"{'Time':<12} {'Emotion':<15} {'Confidence':<12}")
print("="*60)

frame_count = 0
last_emotion = None
last_confidence = None

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("\nWarning: Failed to grab frame from camera")
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

            # Print to terminal only when emotion changes or every 30 frames
            if body_language_class != last_emotion or frame_count % 30 == 0:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"{timestamp:<12} {body_language_class:<15} {max_prob:<12.1%}")
                last_emotion = body_language_class
                last_confidence = max_prob

            # Grab ear coords for label placement
            coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                        , [640,480]).astype(int))

            # Draw emotion label near the ear
            cv2.rectangle(image,
                          (coords[0], coords[1]+5),
                          (coords[0]+len(body_language_class)*20, coords[1]-30),
                          (245, 117, 16), -1)
            cv2.putText(image, body_language_class, coords,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Get status box (top left corner)
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)

            # Display Class
            cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display Probability
            cv2.putText(image, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(max_prob), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            if frame_count == 1:
                print(f"Waiting for face/body detection...")

        cv2.imshow('Body Language Detector - Press Q to Quit', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("\n" + "="*60)
            print("Quitting...")
            break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("DETECTION SUMMARY")
print("="*60)
print(f"Total frames processed: {len(emotions)}")

if emotions:
    emotion_counts = dict((x, emotions.count(x)) for x in set(emotions))
    print(f"\nEmotion distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(emotions)) * 100
        print(f"  {emotion:<15} {count:>5} frames  ({percentage:>5.1f}%)")
else:
    print("No emotions detected")

print("="*60)
