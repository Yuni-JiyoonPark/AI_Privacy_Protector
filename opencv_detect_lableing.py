########################################################################
#############이미지 opencv로 얼굴 감지 후 yolo용으로 라벨링################
########################################################################

import os
import cv2

# Path to dataset
dataset_path = "archive"
subfolders = ["train", "test"]

# Function to detect faces and create YOLO label files
def detect_faces_and_label(subfolder):
    images_path = os.path.join(dataset_path, subfolder, "images")
    labels_path = os.path.join(dataset_path, subfolder, "labels2")
    os.makedirs(labels_path, exist_ok=True)

    for image_file in os.listdir(images_path):
        if image_file.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(images_path, image_file)
            img = cv2.imread(image_path)
            height, width, _ = img.shape
            
            # Convert to grayscale and detect faces
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Create label file
            label_file = os.path.join(labels_path, os.path.splitext(image_file)[0] + ".txt")
            with open(label_file, "w") as f:
                for (x, y, w, h) in faces:
                    # YOLO format: class_id x_center y_center width height (normalized)
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_norm = w / width
                    h_norm = h / height
                    f.write(f"1 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            
            print(f"Processed {image_file}, found {len(faces)} face(s).")

# Process each subfolder
for subfolder in subfolders:
    detect_faces_and_label(subfolder)

print("Labeling completed.")
