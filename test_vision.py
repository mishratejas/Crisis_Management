import cv2
import torch
from ultralytics import YOLO
from agents.vision_agent.vision_agent import analyze_image
from agents.vision_agent.victim_counter import detect_victims_and_vehicles

# ============================================================================
# Configuration
# ============================================================================
IMAGE_PATH = "Gemini_Generated_Image_dbdmcbdbdmcbdbdm.png"
VICTIM_MODEL_PATH = "yolov8n.pt"
DEBRIS_MODEL_PATH = r"agents\vision_agent\best_debris.pt"
OUTPUT_PATH = "image_analysis_complete.jpg"

device = 0 if torch.cuda.is_available() else "cpu"
print(f"[Device] Using: {'GPU' if device == 0 else 'CPU'}")

# ============================================================================
# Load image ONCE with cv2 — same array used for detection AND drawing
# ============================================================================
img_cv = cv2.imread(IMAGE_PATH)
if img_cv is None:
    print(f"❌ Could not load image: {IMAGE_PATH}")
    exit()

img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)  # RGB for YOLO
img_draw = img_cv.copy()                            # BGR copy for drawing

h, w = img_cv.shape[:2]
print(f"📷 Image loaded: {w}x{h} pixels")

# ============================================================================
# Flood & Debris Zone Analysis
# ============================================================================
print("\n" + "="*70)
print("FLOOD & DEBRIS ZONE ANALYSIS")
print("="*70)

result = analyze_image(IMAGE_PATH)

print("\n[Flood & Debris Analysis] Zone Map (severe zones):")
for zone_id, data in result["zone_map"].items():
    if data["severity"] > 0.2:
        print(f"  {zone_id}: flood={data['flood_score']:.3f}, damage={data['damage_score']:.3f}, severity={data['severity']:.3f}")

# ============================================================================
# Load Models
# ============================================================================
print("\n[Models] Loading YOLO models...")

debris_model = YOLO(DEBRIS_MODEL_PATH)
debris_model.to(device)
debris_model.eval()
print(f"  ✅ Debris model loaded: {DEBRIS_MODEL_PATH}")

victim_model = YOLO(VICTIM_MODEL_PATH)
print(f"  ✅ Victim model loaded: {VICTIM_MODEL_PATH}")

# ============================================================================
# Detect People, Animals & Vehicles
# — pass img_rgb directly (NOT load_image) so bbox coords match img_draw
# ============================================================================
print("\n" + "="*70)
print("PEOPLE & VEHICLE DETECTION")
print("="*70)

victims = detect_victims_and_vehicles(img_rgb)

print(f"\n  👤 People:   {victims['people']}")
print(f"  🐾 Animals:  {victims['animals']}")
print(f"  🚗 Vehicles: {victims['vehicles']}")

# ============================================================================
# Detect Debris
# — same img_rgb so bbox coords match img_draw
# ============================================================================
print("\n" + "="*70)
print("DEBRIS DETECTION")
print("="*70)

debris_count = 0
debris_results = debris_model.predict(
    source=img_rgb,
    conf=0.01,
    iou=0.45,
    device=device,
    verbose=False
)

# ============================================================================
# Draw ALL Detections on img_draw
# ============================================================================
print("\n[Visualization] Drawing all detections...")

# ── Debris (GREEN boxes) ────────────────────────────────────────────────────
if debris_results and len(debris_results) > 0:
    result_debris = debris_results[0]
    if result_debris.boxes is not None and len(result_debris.boxes) > 0:
        for box in result_debris.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            debris_count += 1

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img_draw,
                f"Debris {confidence:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

# ── People, Animals, Vehicles ───────────────────────────────────────────────
if victims['detections']:
    for det in victims['detections']:
        x1, y1, x2, y2 = det['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = det['confidence']
        class_name = det['class']

        if class_name == "person":
            color = (255, 0, 0)       # BLUE
            label = f"Person {conf:.2f}"
        elif class_name in ["dog", "cat", "horse", "sheep", "cow",
                            "elephant", "bear", "zebra", "giraffe"]:
            color = (0, 165, 255)     # ORANGE
            label = f"{class_name.capitalize()} {conf:.2f}"
        else:
            color = (0, 0, 255)       # RED for vehicles
            label = f"{class_name.capitalize()} {conf:.2f}"

        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img_draw,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

# ============================================================================
# Save & Summary
# ============================================================================
cv2.imwrite(OUTPUT_PATH, img_draw)

print(f"\n✅ DETECTION SUMMARY:")
print(f"  🟢 Debris:   {debris_count}")
print(f"  🔵 People:   {victims['people']}")
print(f"  🟠 Animals:  {victims['animals']}")
print(f"  🔴 Vehicles: {victims['vehicles']}")
print(f"  📦 Total:    {debris_count + len(victims['detections'])}")
print(f"\n✓ Saved annotated image: {OUTPUT_PATH}")
print(f"📷 Image size: {w}x{h} pixels")
print("\n" + "="*70)