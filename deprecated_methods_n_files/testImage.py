#for debugging trying to get camera to detect
import cv2
from PIL import Image
from ultralytics import YOLO


def test_on_image(image_path, model_path):
    #load the trained YOLO model on the given model_path var
    model = YOLO(model_path)

    #load and preprocess the test image
    frame = cv2.imread(image_path)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (640,640))

    # Perform detection on the image
    results = model(frame_rgb, stream = True)
    for result in results:
        boxes = result.boxes  #boxes object for bounding box outputs
        #masks = result.masks  #masks object for segmentation masks outputs
        keypoints = result.keypoints  #keypoints object for pose outputs
        probs = result.probs  #probs object for classification outputs
        obb = result.obb  #oriented boxes object for OBB outputs
        result.show()  #display to screen
        result.save(filename="result.jpg")  #write to disk




    #display the image with bounding boxes for verification
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img.show()