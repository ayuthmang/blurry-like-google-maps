import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Load a YOLO model trained on license plates/faces

model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
model = YOLO(model_path)

results = model.predict("6th-street-2500x750.jpg", save=True) # saves the result in runs/detect/predict

# Read the image
img = cv2.imread('6th-street-2500x750.jpg')
results = model(img)

for result in results:
    for box in result.boxes:
        # 1. Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # 2. Extract the Region of Interest (ROI)
        roi = img[y1:y2, x1:x2]

        # 3. Apply Gaussian Blur (kernel size must be odd numbers)
        # Higher kernel numbers = heavier blur
        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)

        # 4. Paste the blurred ROI back into the original image array
        img[y1:y2, x1:x2] = blurred_roi

cv2.imwrite('anonymized_traffic_cam.jpg', img)
