import cv2


def preprocess(image_path):
    """
    Preprocess a medical image by applying Gaussian and Median filters.

    Reads the image in grayscale, applies Gaussian blur to reduce
    high-frequency noise, then applies Median blur to remove
    salt-and-pepper noise while preserving edges.

    Args:
        image_path: Path to the input image file.

    Returns:
        Preprocessed grayscale image as a numpy array.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.medianBlur(img, 5)
    return img
