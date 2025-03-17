import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry

# 1. Load the pre-trained SAM model
MODEL_TYPE = "vit_h"  # Choose from "vit_h", "vit_l", or "vit_b"
MODEL_CHECKPOINT = "sam_vit_h_4b8939.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_CHECKPOINT).to(device)
predictor = SamPredictor(sam)

# 2. Load an image
image_path = "open.png"  # Change to your image file
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert OpenCV BGR to RGB

predictor.set_image(image)

# 3. Define a point (manually selected, e.g., on the coffee lid)
input_point = np.array([[400, 150]])  # X, Y coordinate of the object
input_point = np.array([[642, 220]])  # X, Y coordinate of the object
lid_bbox = np.array([191, 111, 526, 219])

input_label = np.array([1])  # 1 = foreground

# 4. Get segmentation mask from SAM
masks, _, _ = predictor.predict(
    point_coords=input_point, 
    point_labels=input_label, 
    multimask_output=False  # Get a single best mask
)

# 5. Display segmentation results
plt.figure(figsize=(10, 5))
plt.imshow(image)
plt.imshow(masks[0], alpha=0.5, cmap="jet")  # Overlay mask
plt.scatter(input_point[:, 0], input_point[:, 1], color="red", s=100, label="Input Point")
plt.legend()
plt.title("Segmented Object")
plt.show()





