from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 model
model = YOLO("models/best4_multi_label/weights/best.pt")

# Load an image
image_path = "dataset/images/train/0.png"  # Change to your image path
frame = cv2.imread(image_path)

# Run YOLOv8 inference
results = model(frame)
print(len(results))
# Draw detections
for result in results:
    for box in result.boxes:
        print(box)
        print('yes')
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        class_id = int(box.cls[0])
        label = f"{model.names[class_id]}: {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the image with detections
cv2.imshow("YOLOv8 Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
