from l2cs import Pipeline
from feat import Detector
import torch
import math
import cv2
import numpy as np
from feat import Detector  # Py-Feat for facial analysis

# Initialize Py-Feat detector
feat_detector = Detector(device="cpu")  # Use 'cuda' if available


def overlay_features(frame, gaze_detector):
    """
    Overlays eye gaze, head pose, facial landmarks and detected emotions on the frame.

    Args:
        frame (numpy.ndarray): Video frame (BGR format).
        gaze_detector (Pipeline): Pretrained gaze detection model.

    Returns:
        numpy.ndarray: Frame with overlays.
    """
    # Extract features
    #print(f"Processing frame with shape: {frame.shape}")

    # Detect faces
    features = feat_detector.detect_facepose(frame)
    faces = features['faces']
    headpose = features['poses']
    #print(f"Detected {len(headpose)} head poses.")
    #print(f"Head pose: {headpose}")
    #print(f"Detected {len(faces)} faces.")
    #print(f"Face bounding boxes: {faces}")
    
    if faces is None or len(faces) == 0:
        return frame
    
    # Detect individual features separately
    landmarks = feat_detector.detect_landmarks(frame, faces)
    #print(f"Detected {len(landmarks)} landmarks.")
    #print(f"Landmarks: {landmarks}")
    
    #aus = feat_detector.detect_aus(frame, landmarks)
    #print(f"Detected {len(aus[0][0])} action units.")
    #print(f"Action units: {aus}")

    emotions = feat_detector.detect_emotions(frame, faces, landmarks)
    emotion_names = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
    #print(f"Detected {len(emotions)} emotions.")
    #print(f"Emotions: {emotions}")
    
    emotions = emotions[0][0]
    #aus = aus[0][0]
    landmarks = landmarks[0][0]
    headpose = headpose[0][0]

    # Get most probable detected emotion
    emotion = np.argmax(emotions)
    # print(f"Length of emotions: {len(emotions)}")
    # print(f"Most probable emotion: {emotion}")
    # print(f"{emotion_names[emotion]}")  
    

    # Draw facial landmarks
    for landmark in landmarks:
        x, y = int(landmark[0]), int(landmark[1])
        cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)  # Light blue for landmarks
    
    # Get head pose
    pose = (headpose[0], headpose[1], headpose[2])
    """
    "left": face[0],
    "top": face[1],
    "right": face[2],
    "bottom": face[3],
    """
    # Get bounding box
    #print(f"Face bounding box: {faces[0][0]}")
    
    x1, y1, x2, y2 = map(int, faces[0][0][:-1])
    tdx, tdy = (x1 + x2) // 2, (y1 + y2) // 2  # Center of face
    size = min(x2 - x1, y2 - y1) // 2  # Scale factor for pose arrows

    # Convert angles to radians
    pitch, roll, yaw = [math.radians(angle) for angle in pose]

    # Compute pose axis endpoints
    x_red = int(size * (math.cos(-yaw) * math.cos(roll)) + tdx)
    y_red = int(size * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(-yaw)) + tdy)

    x_green = int(size * (-math.cos(-yaw) * math.sin(roll)) + tdx)
    y_green = int(size * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(-yaw) * math.sin(roll)) + tdy)

    x_blue = int(size * (math.sin(-yaw)) + tdx)
    y_blue = int(size * (-math.cos(-yaw) * math.sin(pitch)) + tdy)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow box

    # Draw pose arrows
    cv2.arrowedLine(frame, (tdx, tdy), (x_red, y_red), (0, 0, 255), 2, tipLength=0.2)  # Red - Pitch
    cv2.arrowedLine(frame, (tdx, tdy), (x_green, y_green), (0, 255, 0), 2, tipLength=0.2)  # Green - Roll
    cv2.arrowedLine(frame, (tdx, tdy), (x_blue, y_blue), (255, 0, 0), 2, tipLength=0.2)  # Blue - Yaw
    
    """
    # Commented as it increases the lag!
    # Overlay action unit intensities
    au_values = aus.iloc[0].to_dict()
    y_offset = y2 + 20  # Place text below the face
    for au, intensity in au_values.items():
        if intensity > 0:
            text = f"{au}: {intensity:.2f}"
            cv2.putText(frame, text, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 20
    """
    # Display detected emotion
    cv2.putText(frame, f"Emotion: {emotion_names[emotion]}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Overlay gaze detection (Eye gaze estimation)
    gaze_container = gaze_detector.step(frame)
    frame = draw_gaze(frame, gaze_container)
    return frame




def draw_gaze(frame, gaze_container):
    """
    Draw gaze arrow information.
    """
    if len(gaze_container.bboxes) == 0:
        cv2.putText(frame, "No face detected", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    # Parse bounding box
    x_min, y_min, x_max, y_max = [int(coord) for coord in gaze_container.bboxes[0]]

    
    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2


    # Compute sizes
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    # Convert pitch/yaw to radians if they are given in degrees
    eye_pitch_radians = gaze_container.pitch[0]
    eye_yaw_radians = gaze_container.yaw[0]
    #eye_pitch = math.radians(eye_pitch_degrees)
    #eye_yaw = math.radians(eye_yaw_degrees)

    # Compute gaze direction
    #arrow_length = 1000
    #dx = int(-arrow_length * math.sin(eye_yaw) * math.cos(eye_pitch))
    #dy = int(-arrow_length * math.sin(eye_pitch))
    dx = -bbox_width * np.sin(eye_pitch_radians) * np.cos(eye_yaw_radians)
    dy = -bbox_width * np.sin(eye_yaw_radians)
    
    # Draw gaze direction as an arrow
    cv2.arrowedLine(
        frame,
        (cx,cy),
        (np.round(cx + dx).astype(int), np.round(cy + dy).astype(int)),
        (0, 0, 255),
        2,
        cv2.LINE_AA,
        tipLength=0.18,
    )
    
    # # Label with pitch/yaw
    # label = f"yaw {eye_yaw_radians:.2f}  pitch {eye_pitch_radians:.2f}"
    # cv2.putText(
    #     frame,
    #     label,
    #     (x_min, y_min - 10),
    #     cv2.FONT_HERSHEY_PLAIN,
    #     1.5,
    #     (255, 0, 0),
    #     2,
    # )

    return frame



def real_time_overlay(detector):
    """
    Runs real-time video capture with overlaid Py-Feat features and L2CS Eyegaze.
    """
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("Error: Unable to access camera.")
        return

    print("Starting real-time feature overlay... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        try:
            frame = overlay_features(frame, detector)
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

        # Display frame
        cv2.imshow("Real-Time Py-Feat Overlay", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Initialize the gaze pipeline
model_path = r"models/L2CSNet_gaze360.pkl"  # Update model path
gaze_detector = Pipeline(weights=model_path, arch='ResNet50', device=torch.device('cpu'))

real_time_overlay(gaze_detector)
