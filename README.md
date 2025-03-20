# Action Sequence Generation for Coffee Making Video

This repository is designed for generating action sequences from a coffee-making video. It provides tools to train and fine-tune a machine learning model for detecting objects and movements during the coffee-making process.

### Customization for Your Setup

If you are working with a custom coffee machine setup, the model needs to be fine-tuned to ensure stability and robustness. Specifically, it should be adjusted to your personal setup, including:

- The lid and start button of your coffee machine
- The ground coffee and paper filter you're using

### Preparing the Training Data

To train the model, you need only 20-30 labeled images. The following scripts help with this process:

1. **`screenshot.py`**: Automatically generates screenshots from your video, one per second.
2. **`data_labeling.py`**: Provides a user interface for labeling each screenshot, specifically to define bounding boxes around each object in the video.
3. **`data_split.py`**: Splits the labeled data into training and validation datasets.

### Fine-Tuning the Model

Once your data is prepared, use the following command to fine-tune the model:

```bash
yolo task=detect mode=train model=best.pt data=dataset/data.yaml epochs=50 imgsz=640 project=models name=custom_model
```

The trained model will be saved to `custom_model/weights/best.pt`.

### Generating Action Sequences

The **`action_rego.py`** script utilizes the trained object detection model and hand detection to output:

- Movement events
- Object position history
- Grasping points history
- Time when the start button is pressed

The **`action_seq.py`** script will then display the video along with the recognized actions.

### Customization for Different Resolutions

Note: If you're using a different resolution or video speed compared to the original (v3.MOV), you may need to adjust the following parameters:

- **`movement_threshold`**: To detect when objects are moving.
- **`glittering_threshold`**: To prevent misclassifications from affecting the movement events.

