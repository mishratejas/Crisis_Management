import torch
from ultralytics import YOLO

# Device detection
device = 0 if torch.cuda.is_available() else "cpu"

# Load COCO-trained YOLO model (detects people, animals, vehicles)
model = YOLO("yolov8n.pt")
model.to(device)
model.eval()

# COCO class names relevant to disasters
DISASTER_CLASSES = {
    0: "person",        # People
    16: "dog",          # Animals
    17: "cat",
    18: "horse",
    19: "sheep",
    20: "cow",
    21: "elephant",
    22: "bear",
    23: "zebra",
    24: "giraffe",
    # Vehicles
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
}


def detect_victims_and_vehicles(image):
    """
    Detect people, animals, and vehicles in image using YOLO.
    
    Args:
        image: numpy array (RGB format)
    
    Returns:
        Dict with counts and detections:
        {
            "people": int,
            "animals": int, 
            "vehicles": int,
            "detections": [{"class": str, "conf": float, "bbox": [x1,y1,x2,y2]}, ...]
        }
    """
    
    if image is None:
        return {"people": 0, "animals": 0, "vehicles": 0, "detections": []}
    
    try:
        # Run YOLO inference
        results = model.predict(
            source=image,
            conf=0.15,  # Lowered from 0.40 for better detection
            device=device,
            verbose=False
        )
        
        people_count = 0
        animals_count = 0
        vehicles_count = 0
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Check if it's a person
                    if class_id == 0:  # person
                        people_count += 1
                        class_name = "person"
                    # Check if it's an animal
                    elif class_id in [16, 17, 18, 19, 20, 21, 22, 23, 24]:
                        animals_count += 1
                        class_name = DISASTER_CLASSES.get(class_id, f"animal_{class_id}")
                    # Check if it's a vehicle
                    elif class_id in [2, 3, 4, 5, 6, 7, 8]:
                        vehicles_count += 1
                        class_name = DISASTER_CLASSES.get(class_id, f"vehicle_{class_id}")
                    else:
                        continue  # Skip other classes
                    
                    detections.append({
                        "class": class_name,
                        "confidence": round(confidence, 3),
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
                    })
        
        result_dict = {
            "people": people_count,
            "animals": animals_count,
            "vehicles": vehicles_count,
            "detections": detections
        }
        
        print(f"[Victim Counter] 👤 People: {people_count}, 🐾 Animals: {animals_count}, 🚗 Vehicles: {vehicles_count}")
        
        return result_dict
    
    except Exception as e:
        print(f"[Victim Counter] Error: {e}")
        return {"people": 0, "animals": 0, "vehicles": 0, "detections": []}


def count_victims_by_zone(detections, image, grid_size=10):
    """
    Count victims (people) in each zone.
    Kept for backward compatibility.
    
    Args:
        detections: List of detection dicts
        image: numpy array
        grid_size: zones per side (10 = 100 zones)
    
    Returns:
        Dict: zone -> victim count
    """

    height, width = image.shape[:2]

    cell_w = width // grid_size
    cell_h = height // grid_size

    victim_map = {}

    for det in detections:

        # filter only people
        if det.get("class") != "person":
            continue

        # ignore low confidence
        if det.get("confidence", 0) < 0.4:
            continue

        x1, y1, x2, y2 = det["bbox"]

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        gx = cx // cell_w
        gy = cy // cell_h

        zone = (gy, gx)

        if zone not in victim_map:
            victim_map[zone] = 0

        victim_map[zone] += 1

    return victim_map