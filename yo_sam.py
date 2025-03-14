from ultralytics import YOLO
import cv2
import numpy as np

# Load pre-trained YOLOv8 model
model = YOLO("best.pt")  # Use "yolov8n.pt" (nano) for fast detection

# Read image
image_path = "coffee_maker.jpg"
image = cv2.imread(image_path)

# Run YOLO on the image
results = model(image)

# Extract bounding boxes (assume class 0 = coffee maker if trained)
for box in results[0].boxes.xyxy:  # xyxy format: (x1, y1, x2, y2)
    x1, y1, x2, y2 = map(int, box[:4])
    bounding_box = np.array([x1, y1, x2, y2])
    break  # Use first detection


from segment_anything import sam_model_registry, SamPredictor

# Load SAM model (choose "vit_h", "vit_l", or "vit_b" depending on size)
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Download from Meta AI's GitHub
model_type = "vit_h"

# Initialize SAM
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# Convert image to RGB (SAM requires RGB format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run SAM
predictor.set_image(image_rgb)
masks, _, _ = predictor.predict(box=bounding_box[None, :], multimask_output=False)

# Save mask
np.savez("saved_mask.npz", mask=masks[0])


# Load bounding box (optional)
# bounding_box = np.load("saved_bbox.npz")["bbox"]

# Load mask
loaded_data = np.load("saved_mask.npz")
saved_mask = loaded_data["mask"]

import matplotlib.pyplot as plt

# Overlay mask on image
masked_image = image_rgb.copy()
masked_image[saved_mask == 0] = [0, 0, 0]  # Black out background

plt.imshow(masked_image)
plt.axis("off")
plt.show()


