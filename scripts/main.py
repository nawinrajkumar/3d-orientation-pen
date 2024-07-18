import cv2
from ultralytics import YOLO

# Load the model
model = YOLO("C:/Users/Nawin/Projects/Hand-Orientation/models/best.pt")

# Load the video
cap = cv2.VideoCapture("C:/Users/Nawin/Projects/Hand-Orientation/input.mp4")

while True:
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    resized_frame = cv2.resize(frame, (640, 480))

    # Make the prediction
    results = model.predict(resized_frame)

    for result in results:
        keypoints = result.keypoints.xy[0]

        # Get the first and second keypoints
        first_point = keypoints[0].cpu().numpy()
        second_point = keypoints[1].cpu().numpy()

        # Draw circles around the keypoints
        cv2.circle(resized_frame, 
                   (int(first_point[0]), int(first_point[1])), 
                   5, (0, 255, 0), -1)
        
        cv2.circle(resized_frame, 
                   (int(second_point[0]), int(second_point[1])),
                    5, (0, 255, 0), -1)

        # Draw a line between the keypoints
        cv2.line(resized_frame, (int(first_point[0]), int(first_point[1])),
                  (int(second_point[0]), int(second_point[1])), (255, 0, 0), 2)
        
    # Display the frame
    cv2.imshow("frame", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
