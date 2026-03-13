import cv2

def load_image(image_path, max_size=1024):
    """
    Load and preprocess image for the Vision Agent
    """

    # Load image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    # Convert BGR → RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]

    # Resize if image is too large
    if max(height, width) > max_size:

        scale = max_size / max(height, width)

        new_w = int(width * scale)
        new_h = int(height * scale)

        image = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

    return image