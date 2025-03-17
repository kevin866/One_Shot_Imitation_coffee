

from ultralytics import YOLO
import cv2
import numpy as np

from segment_anything import sam_model_registry, SamPredictor

# Load SAM model (choose "vit_h", "vit_l", or "vit_b" depending on size)
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Download from Meta AI's GitHub
model_type = "vit_h"
image_path = "open.png"
image = cv2.imread(image_path)

# Initialize SAM
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# Convert image to RGB (SAM requires RGB format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # Manually set a bounding box around the lid (adjust these values)
lid_bbox = np.array([191, 111, 526, 219])
lid_bbox = np.array([448,401,825,14])
# Run SAM
predictor.set_image(image_rgb)
# Run SAM on the lid bounding box
masks, _, _ = predictor.predict(box=lid_bbox[None, :], multimask_output=False)

# Save the lid mask
np.savez("saved_lid_mask.npz", mask=masks[0])
import matplotlib.pyplot as plt

# Load mask
lid_mask_data = np.load("saved_lid_mask.npz")
lid_mask = lid_mask_data["mask"]
# lid_mask = masks[0]["mask"]
# Overlay mask on image
lid_segmented = image_rgb.copy()
lid_segmented[lid_mask == 0] = [0, 0, 0]  # Black out background

plt.imshow(lid_segmented)
plt.axis("off")
plt.title("Lid Segmentation")
plt.show()