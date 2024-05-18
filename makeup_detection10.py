from ultralytics import YOLO
import cv2
import time
from ultralytics.utils.plotting import Annotator

# Load the YOLO trained models for makeup and no makeup
model = YOLO("Facial Makeup Detection using YOLOv8/runs/detect/train/weights/best.pt")

# Check if the camera is opened successfully
cam = cv2.VideoCapture(0)  # 0 for default webcam
if not cam.isOpened():
    raise ("No camera")

# Read frame from webcam
while True:
    ret, image = cam.read()
    result = model(image, stream=True)

    # Draw boxes on the image
    for r in result:
        # Initialize annotator object
        annotator = Annotator(image)

        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # Get bounding box coordinates
            c = int(box.cls)  # Get class label
            conf = box.conf[0]  # Get confidence score

            # Create label with class name and confidence score
            label = f"{model.names[c]} {conf:.2f}"

            # Draw the bounding box with the label
            annotator.box_label(b, label)

    image = annotator.result()

    # Display the result
    cv2.imshow("Facial Makeup Detection", image)

    # Exit loop
    print("time", time.time())
    if cv2.waitKey(30) == 27:  # 27 is esc key in ASCII table
        break

cam.release()
cv2.destroyAllWindows()