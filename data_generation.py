import cv2
import os

# Define dataset paths
dataset_dir = "dataset"
images_dir = os.path.join(dataset_dir, "images/train")  # Change to "val" if needed
labels_dir = os.path.join(dataset_dir, "labels/train")  # Change to "val" if needed
yaml_path = os.path.join(dataset_dir, "data.yaml")

# Create dataset folders if they don't exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Folder containing images
image_folder = "image"
image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")]

# Define classes
classes = ["lid", "start", "coffee_bean"]

def draw_bbox(event, x, y, flags, param):
    global bbox, drawing, img
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox = [(x, y)]
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_copy = img.copy()
        cv2.rectangle(img_copy, bbox[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("Draw Bounding Box", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        bbox.append((x, y))
        drawing = False
        cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 2)
        cv2.imshow("Draw Bounding Box", img)

def convert_bbox(img_shape):
    h, w = img_shape[:2]
    x_min, y_min = bbox[0]
    x_max, y_max = bbox[1]
    x_center = (x_min + x_max) / 2 / w
    y_center = (y_min + y_max) / 2 / h
    width = (x_max - x_min) / w
    height = (y_max - y_min) / h
    return x_center, y_center, width, height
for i in range(len(classes)):
    for image_name in image_files:
        print(f"current label {classes[i]}")
        image_path = os.path.join(image_folder, image_name)
        new_image_name = f"{image_name.replace('.png', '_')}{classes[i]}.png"
        img = cv2.imread(image_path)
        if img is None:
            print(f"Skipping {image_name}: Unable to load image")
            continue

        # Save image without bounding box
        original_image_path = os.path.join(images_dir, new_image_name)

        cv2.imwrite(original_image_path, img)

        bbox = []
        drawing = False
    
        cv2.imshow("Draw Bounding Box", img)
        cv2.setMouseCallback("Draw Bounding Box", draw_bbox)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(bbox) == 2:
            new_image_path = os.path.join(images_dir, new_image_name)
            cv2.imwrite(new_image_path, img)

            bbox_data = [abs(i) for i in convert_bbox(img.shape)]

            # Ask user to label as "lid" or "start"
            class_label = i
            label_name = new_image_name.replace(".jpg", ".txt").replace(".png", ".txt")
            label_path = os.path.join(labels_dir, label_name)
            with open(label_path, "w") as f:
                f.write(f"{class_label} {bbox_data[0]} {bbox_data[1]} {bbox_data[2]} {bbox_data[3]}\n")
            
            print(f"Image saved to: {new_image_path}")
            print(f"Label saved to: {label_path}")
            print(f"Original image (no bbox) saved to: {original_image_path}")
            
    

# Generate data.yaml
yaml_content = f"""
train: images/train
val: images/val

nc: {len(classes)}
names: {classes}
"""

with open(yaml_path, "w") as f:
    f.write(yaml_content.strip())

print(f"Dataset structure created at: {dataset_dir}")
