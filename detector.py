from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path="yolov8l.pt", conf_thresh=0.3):
        self.model = YOLO(model_path)  # Load YOLOv8 model
        self.conf_thresh = conf_thresh  # Minimum confidence threshold

        # Allowed vehicle classes (COCO IDs)
        self.vehicle_classes = {
            1: "bicycle",     
            2: "car",
            3: "motorcycle",
            7: "truck"
        }

    def detect(self, image):
        results = self.model.predict(image, imgsz=928)  # Run YOLOv8 inference
        detections = []

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls)  # Get predicted class ID
                conf = float(box.conf)   # Get confidence score

                # Skip bus predictions (COCO class ID 5)
                if class_id == 5:
                    continue  # Ignore buses completely

                if conf >= self.conf_thresh and class_id in self.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coords
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "class": self.vehicle_classes[class_id],  # Vehicle type
                        "conf": conf  # Confidence
                    })

        return detections  # Return all valid detections

