from ultralytics import YOLO

# load model once
model = YOLO("yolov8n.pt")

def detect_objects(image):

    results = model(image)

    detections = []

    for r in results:

        for box in r.boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "class_id": class_id
            })

    return detections