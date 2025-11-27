import mediapipe as mp
import cv2
import pickle
import pandas as pd
import numpy as np
import warnings

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

print("="*70)
print("DEBUG MODE - Body Language Detector")
print("="*70)
print("This will show you what MediaPipe is detecting\n")

frame_count = 0
detection_count = 0

# Initiate holistic model with lower confidence thresholds
with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
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

        # Debug: Check what's detected
        if frame_count % 30 == 0:
            print(f"\nFrame {frame_count}:")
            print(f"  Face detected: {results.face_landmarks is not None}")
            print(f"  Pose detected: {results.pose_landmarks is not None}")
            print(f"  Left hand detected: {results.left_hand_landmarks is not None}")
            print(f"  Right hand detected: {results.right_hand_landmarks is not None}")

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

        # Try to make predictions
        if results.pose_landmarks and results.face_landmarks:
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

                if frame_count % 10 == 0:
                    print(f"\n✓ DETECTED: {body_language_class} ({max_prob:.1%})")

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

            except Exception as e:
                if frame_count % 30 == 0:
                    print(f"Error making prediction: {e}")

        else:
            # Show what's missing
            missing = []
            if not results.pose_landmarks:
                missing.append("POSE")
            if not results.face_landmarks:
                missing.append("FACE")

            if missing:
                cv2.putText(image, f'Missing: {", ".join(missing)}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'Move closer / better lighting', (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Debug: Body Language Detector (Press Q to quit)', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("\n\nStopping...")
            break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total frames: {frame_count}")
print(f"Successful detections: {detection_count}")

if emotions:
    emotion_counts = dict((x, emotions.count(x)) for x in set(emotions))
    print(f"\nDetected emotions:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {emotion}: {count} times")
else:
    print("\nNo detections made.")
    print("Tips:")
    print("  - Make sure your face and upper body are fully visible")
    print("  - Move closer to the camera")
    print("  - Ensure good lighting")
    print("  - Face the camera directly")

print("="*70)
