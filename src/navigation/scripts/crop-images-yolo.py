import os
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Define directories
input_dir = "target_image"
output_dir = "cropped_images_target"
os.makedirs(output_dir, exist_ok=True)

# Process images
results = model.predict(source=input_dir, stream=True)

for result in results:
    # Get original image path and name
    image_path = result.path  # Updated from result.files[0]
    image_name = os.path.basename(image_path)
    
    # Load original image
    img = cv2.imread(image_path)
    
    # Process each detection
    for j, box in enumerate(result.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box[:4])
        crop = img[y1:y2, x1:x2]
        
        # Save cropped image
        cv2.imwrite(
            os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_crop_{j}.jpg"),
            crop
        )
