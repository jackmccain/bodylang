from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import pickle
import numpy as np
import pandas as pd
import warnings
import json
import base64

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the model
with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables
current_emotion = "Waiting..."
current_confidence = 0.0
emotion_buffer = []  # For temporal smoothing
BUFFER_SIZE = 1  # No averaging - immediate predictions

# Initialize holistic model once for reuse
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1)

def smooth_predictions(emotion, confidence):
    """Apply temporal smoothing to reduce jitter"""
    global emotion_buffer

    emotion_buffer.append((emotion, confidence))

    # Keep only last BUFFER_SIZE predictions
    if len(emotion_buffer) > BUFFER_SIZE:
        emotion_buffer.pop(0)

    # Count emotion occurrences and average confidence
    emotion_counts = {}
    emotion_confidences = {}

    for emo, conf in emotion_buffer:
        emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
        if emo not in emotion_confidences:
            emotion_confidences[emo] = []
        emotion_confidences[emo].append(conf)

    # Get most common emotion
    smoothed_emotion = max(emotion_counts, key=emotion_counts.get)
    smoothed_confidence = np.mean(emotion_confidences[smoothed_emotion])

    return smoothed_emotion, smoothed_confidence

def process_frame_data(frame):
    """Process a single frame and return emotion detection results"""
    global current_emotion, current_confidence

    try:
        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(80, 110, 10), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(80, 256, 121), thickness=1))

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(245, 117, 66), thickness=2, circle_radius=4),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(245, 66, 230), thickness=2))

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(121, 22, 76), thickness=2, circle_radius=4),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(121, 44, 250), thickness=2))

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(80, 22, 10), thickness=2, circle_radius=4),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(80, 44, 121), thickness=2))

        # Make prediction
        if results.pose_landmarks and results.face_landmarks:
            try:
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                         for landmark in pose]).flatten())

                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                         for landmark in face]).flatten())

                row = pose_row + face_row
                X = pd.DataFrame([row])

                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                confidence = round(float(np.max(body_language_prob)), 3)

                # Apply temporal smoothing
                smoothed_emotion, smoothed_confidence = smooth_predictions(
                    body_language_class, confidence)

                current_emotion = smoothed_emotion
                current_confidence = smoothed_confidence

            except Exception as e:
                current_emotion = "Processing..."
                current_confidence = 0.0
        else:
            current_emotion = "Face not detected"
            current_confidence = 0.0

        # Encode processed frame back to base64
        _, buffer = cv2.imencode('.jpg', image)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            'frame': frame_base64,
            'emotion': current_emotion,
            'confidence': current_confidence
        }

    except Exception as e:
        print(f"Error processing frame: {e}")
        return {
            'frame': None,
            'emotion': 'Error',
            'confidence': 0.0
        }

def generate_frames_legacy():
    """Legacy function for local testing only"""
    global current_emotion, current_confidence

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1) as holistic_local:

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Process with MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic_local.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks with subtle styling
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(80, 110, 10), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(80, 256, 121), thickness=1))

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(245, 117, 66), thickness=2, circle_radius=4),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(245, 66, 230), thickness=2))

            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(121, 22, 76), thickness=2, circle_radius=4),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(121, 44, 250), thickness=2))

            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(80, 22, 10), thickness=2, circle_radius=4),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(80, 44, 121), thickness=2))

            # Make prediction
            if results.pose_landmarks and results.face_landmarks:
                try:
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                             for landmark in pose]).flatten())

                    face = results.face_landmarks.landmark
                    face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                             for landmark in face]).flatten())

                    row = pose_row + face_row
                    X = pd.DataFrame([row])

                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
                    confidence = round(float(np.max(body_language_prob)), 3)

                    # Apply temporal smoothing
                    smoothed_emotion, smoothed_confidence = smooth_predictions(
                        body_language_class, confidence)

                    current_emotion = smoothed_emotion
                    current_confidence = smoothed_confidence

                except Exception as e:
                    current_emotion = "Processing..."
                    current_confidence = 0.0
            else:
                current_emotion = "Face not detected"
                current_confidence = 0.0

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Legacy endpoint for local testing"""
    return Response(generate_frames_legacy(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a frame sent from the browser"""
    try:
        data = request.get_json()
        frame_data = data.get('frame')

        if not frame_data:
            return jsonify({'error': 'No frame data'}), 400

        # Decode base64 image
        frame_data = frame_data.split(',')[1] if ',' in frame_data else frame_data
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid frame'}), 400

        # Process frame
        result = process_frame_data(frame)

        return jsonify(result)

    except Exception as e:
        print(f"Error in process_frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/emotion_data')
def emotion_data():
    return jsonify({
        'emotion': current_emotion,
        'confidence': current_confidence
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))

    print("\n" + "="*60)
    print("Body Language Detector - Web Interface")
    print("="*60)
    print("Starting server...")
    print(f"Open your browser and go to: http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")

    app.run(debug=False, threaded=False, host='0.0.0.0', port=port, use_reloader=False)
