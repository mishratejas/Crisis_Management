from .preprocess import load_image
from .detector import detect_objects
from .flood_segmentation import detect_flood
from .grid_mapper import build_heatmap
from .severity import classify_severity
from .victim_counter import count_victims_by_zone


def analyze_image(image_path):

    image = load_image(image_path)

    detections = detect_objects(image)

    flood_mask = detect_flood(image)

    heatmap = build_heatmap(image, detections, flood_mask)

    severity = classify_severity(heatmap)

    victims = count_victims_by_zone(detections, image)

    return {
        "heatmap": heatmap.tolist(),
        "severity": severity,
        "victims": victims
    }