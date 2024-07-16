import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing for visualizing landmarks.
mp_drawing = mp.solutions.drawing_utils

# Capture video from webcam.
cap = cv2.VideoCapture(0)

def calculate_angles(wrist, index_mcp, pinky_mcp):
    # Calculate vectors from wrist to index and pinky MCP joints.
    vector1 = np.array([index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z])
    vector2 = np.array([pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z])
    
    # Calculate the normal to the plane defined by the vectors (cross product).
    normal = np.cross(vector1, vector2)
    
    # Normalize the normal vector.
    normal = normal / np.linalg.norm(normal)
    
    # Calculate roll, pitch, yaw.
    roll = np.arctan2(normal[1], normal[2])
    pitch = np.arctan2(-normal[0], np.sqrt(normal[1]**2 + normal[2]**2))
    yaw = np.arctan2(vector1[1], vector1[0])
    
    # Convert radians to degrees.
    roll = np.degrees(roll)
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    
    return roll, pitch, yaw

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find hands.
    results = hands.process(rgb_frame)

    # Draw hand landmarks and calculate 3D orientation of the wrist.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks.
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates for the wrist, index finger MCP, and pinky MCP.
            landmarks = hand_landmarks.landmark
            wrist = landmarks[mp_hands.HandLandmark.WRIST]
            index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]

            # Calculate roll, pitch, yaw.
            roll, pitch, yaw = calculate_angles(wrist, index_mcp, pinky_mcp)
            print(f'Roll: {roll:.2f}°, Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°')

            # Draw a circle on the wrist position for visualization.
            h, w, _ = frame.shape
            wrist_pixel_coords = (int(wrist.x * w), int(wrist.y * h))
            cv2.circle(frame, wrist_pixel_coords, 5, (0, 255, 0), -1)

    # Display the frame.
    cv2.imshow('Hand Pose Estimation', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
