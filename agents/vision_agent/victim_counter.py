def count_victims_by_zone(detections, image, grid_size=10):

    height, width = image.shape[:2]

    cell_w = width // grid_size
    cell_h = height // grid_size

    victim_map = {}

    for det in detections:

        # filter only people
        if det["class_id"] != 0:
            continue

        # ignore low confidence
        if det["confidence"] < 0.4:
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