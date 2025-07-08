import imageio.v3 as iio
import cv2
import os
from detector import VehicleDetector
from utils import draw_boxes

# Initialize detector with confidence threshold
detector = VehicleDetector(conf_thresh=0.3)

# Counters for vehicles
total_counts = {"car": 0, "motorcycle": 0, "truck": 0, "bicycle": 0}

# Input/output folders
input_folder = "data/test_images/"
output_folder = "output/processed_images/"
os.makedirs(output_folder, exist_ok=True)

# Process each image
for image_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, image_name)

    if not os.path.isfile(img_path):
        print(f"❌ Skipping non-file: {img_path}")
        continue

    try:
        # Read and process image
        img = iio.imread(img_path)
        if img is None:
            print(f"❌ Could not read image: {img_path}")
            continue

        # Detect vehicles
        detections = detector.detect(img)
        for det in detections:
            total_counts[det["class"]] += 1

        # Draw boxes and save
        annotated = draw_boxes(img, detections)
        out_path = os.path.join(output_folder, image_name)
        cv2.imwrite(out_path, annotated)
        
        print(f"✔ Processed: {image_name} ({len(detections)} vehicles detected)")

    except Exception as e:
        print(f"❌ Error processing {image_name}: {str(e)}")
        continue

# Print final counts
print("\n=== ✅ Vehicle Counts ===")
for vehicle_type, count in total_counts.items():
    print(f"{vehicle_type}: {count}")



