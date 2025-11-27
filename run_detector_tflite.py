import mediapipe as mp
import cv2
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

# Load the TensorFlow Lite model
print("Loading TensorFlow Lite model...")
interpreter = tf.lite.Interpreter(model_path='body_language.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded successfully!\n")

# Class labels (as used in training)
class_labels = ['Angry', 'Confused', 'Depressed', 'Excited', 'Happy', 'Pain', 'Sad', 'Surprised', 'Tension']

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
emotions = []
warnings.filterwarnings('ignore')

print("="*70)
print("BODY LANGUAGE DETECTOR - TensorFlow Lite Model")
print("="*70)
print("Detecting emotions in real-time...")
print("Press 'Q' in the video window to quit\n")
print("="*70)
print(f"{'Time':<12} {'Emotion':<15} {'Confidence':<12}")
print("="*70)

frame_count = 0
detection_count = 0

# Initiate holistic model
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

        # Make predictions
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

                # Prepare input for TFLite model
                input_data = np.array([row], dtype=np.float32)

                # Set the input tensor
                interpreter.set_tensor(input_details[0]['index'], input_data)

                # Run inference
                interpreter.invoke()

                # Get the output
                output_data = interpreter.get_tensor(output_details[0]['index'])

                # Get predicted class and probability
                predicted_class_idx = np.argmax(output_data[0])
                body_language_class = class_labels[predicted_class_idx]
                max_prob = round(float(np.max(output_data[0])), 3)

                emotions.append(str(body_language_class))
                detection_count += 1

                # Print to terminal every 10 frames
                if frame_count % 10 == 0:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"{timestamp:<12} {body_language_class:<15} {max_prob:<12.1%}")

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
                if frame_count % 30 == 1:
                    print(f"Error during prediction: {e}")

        else:
            # Show what's missing
            missing = []
            if not results.pose_landmarks:
                missing.append("POSE")
            if not results.face_landmarks:
                missing.append("FACE")

            if missing and frame_count % 30 == 0:
                print(f"Frame {frame_count}: Missing {', '.join(missing)}")

        cv2.imshow('Body Language Detector (TFLite) - Press Q to quit', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("\n\nStopping...")
            break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print("DETECTION SUMMARY")
print("="*70)
print(f"Total frames: {frame_count}")
print(f"Successful detections: {detection_count}")

if emotions:
    emotion_counts = dict((x, emotions.count(x)) for x in set(emotions))
    print(f"\nEmotion distribution:")
    print("="*70)
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(emotions)) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {emotion:<15} {count:>5} ({percentage:>5.1f}%)  {bar}")
    print(f"\nMost detected: {max(emotion_counts, key=emotion_counts.get)}")
else:
    print("No detections made")

print("="*70)
