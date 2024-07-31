## Bayraktar TB2 Detection with YOLOv4

### YOLO

YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system. It frames object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities.

Key Features of YOLO:

* Speed: YOLO is incredibly fast because it only looks at the image once (hence the name) to predict what objects are present and where they are.
* Unified Architecture: It uses a single neural network to perform classification and localization tasks simultaneously.
* High Accuracy: Despite its speed, YOLO maintains high accuracy, making it suitable for real-time applications.
* Generalization: YOLO generalizes well to new domains, which makes it an effective solution for a variety of detection tasks.

YOLO divides the image into a grid and predicts bounding boxes and probabilities for each grid cell. This approach allows YOLO to understand the global context of the image, making it particularly effective for detecting large objects or groups of objects.

### Model



### How to Run: 

1. [YOLO model output](https://drive.google.com/file/d/1MpWDwkxqJroh_HDa04smDY_wF8Cb8fWL/view?usp=sharing) , Download the yolo model output resulting from the training from this link and convert it to zip format and upload it to your drive account along with the files in my github account.


2. Open Google Colab in your browser and connect Google Colab to your drive account


3. [tbiki_detection.ipynb](https://github.com/rmmehmet/bayraktar-tb2-detection/blob/main/tbiki_yolo_model/source_code/tbiki_detection.ipynb) , Download this colab file and connect it to your google drive account after opening it

```python
cfg_path =   "/content/drive/MyDrive/tbiki_yolo_model/yolov4/darknet/tbiki_yolov4.cfg" # cfg path

weights_path = "/content/drive/MyDrive/tbiki_yolov4_best.weights" # weights path

video_path = "/content/drive/MyDrive/tb2_video.mp4"  # video path
```
4. [bayraktar tb2 test video](https://github.com/rmmehmet/bayraktar-tb2-detection/blob/main/tbiki_yolo_model/yolov4/darknet/tb2_video.mp4) , [tbiki_yolov4.cfg](https://github.com/rmmehmet/bayraktar-tb2-detection/blob/main/tbiki_yolo_model/yolov4/darknet/tbiki_yolov4.cfg) , [YOLO model output (tbiki_yolov4_best.weights)](https://drive.google.com/file/d/1MpWDwkxqJroh_HDa04smDY_wF8Cb8fWL/view?usp=sharing) update the path to these files in the code block in tbiki_detection.ipynb shown above

5. After these steps, run the tbiki_detection.ipynb file in google colab

### Example Detection

![sample detection on test video
](tbiki_yolo_model\tb2_output-2.png)

### Graphic From the Training Process

![Training Process 1](tbiki_yolo_model\train-1.png)
![Training Process 1](tbiki_yolo_model\train-2.png)
![Training Process 1](tbiki_yolo_model\train-3.png)
![Training Process 1](tbiki_yolo_model\train-4.png)
![Training Process 1](tbiki_yolo_model\train-5.png)
![Training Process 1](tbiki_yolo_model\train-6.png)

### Algorithm Created for Detection

```python
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 3 01:46:23 2024
@author: Mehmet
"""

import cv2
import numpy as np
from google.colab.patches import cv2_imshow

cfg_path =   "/content/drive/MyDrive/tbiki_yolo_model/yolov4/darknet/tbiki_yolov4.cfg" # cfg path
weights_path = "/content/drive/MyDrive/tbiki_yolov4_best.weights" # weights path


video_path = "/content/drive/MyDrive/tb2_video.mp4"  # video path
cap = cv2.VideoCapture(video_path)

# load YOLOv4 model
model = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

# class label
labels = ["Bayraktar | TB2"]

# colors
colors = [(0, 255, 255), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 0)]

# input size of the model
input_width = 416
input_height = 416

# loop on video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # resize the input frame to the appropriate size
    blob = cv2.dnn.blobFromImage(frame, 1/255, (input_width, input_height), swapRB=True, crop=False)

    # set the model's input
    model.setInput(blob)

    # removing the fixing layers
    output_layers_names = model.getUnconnectedOutLayersNames()
    layer_outputs = model.forward(output_layers_names)

    # processing of detections
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # confidence threshold
            if confidence > 0.5:
                # obtaining information about the detected object
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                # calculating the corner coordinates of the rectangle
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, labels[class_ids[i]], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # showing the video
    cv2_imshow(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

```


