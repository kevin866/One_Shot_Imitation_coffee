import os

def make_values_positive(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        with open(file_path, 'w') as file:
            for line in lines:
                values = line.split()
                positive_values = [str(abs(float(val))) for val in values]
                file.write(" ".join(positive_values) + "\n")

# Apply to both train and val folders
make_values_positive("dataset/labels/train")
make_values_positive("dataset/labels/val")
