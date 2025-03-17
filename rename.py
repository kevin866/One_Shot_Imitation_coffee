import os

# Define dataset paths
dataset_path = "dataset"
image_dirs = [os.path.join(dataset_path, "images", "train"), os.path.join(dataset_path, "images", "val")]
label_dirs = [os.path.join(dataset_path, "labels", "train"), os.path.join(dataset_path, "labels", "val")]

# Function to rename files in a directory
def rename_files(directory, file_extension):
    i = 0
    for filename in os.listdir(directory):
        if filename.endswith(file_extension):
            old_path = os.path.join(directory, filename)
            new_filename = str(i)+".png"  # Remove first 8 characters
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
            i = i+1
image = "image"
# Rename images and labels
rename_files(image, ".png")  # Change if using PNG

print("Renaming completed!")
