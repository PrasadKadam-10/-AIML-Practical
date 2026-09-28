import cv2
from ultralytics import YOLO

# Load the pre-trained YOLO model
model = YOLO("yolo11n.pt")

# Open the CCTV video
cap = cv2.VideoCapture("crosswalk.avi")

while cap.isOpened():

    # Read one frame
    ret, frame = cap.read()

    # Stop when video ends
    if not ret:
        break

    # Detect objects
    results = model(frame)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Display the video
    cv2.imshow("CCTV Object Detection", annotated_frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video
cap.release()
cv2.destroyAllWindows()
