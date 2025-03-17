import matplotlib.pyplot as plt

from ultralytics import YOLO
import cv2
import numpy as np

# Load mask
lid_mask_data = np.load("saved_lid_mask.npz")
