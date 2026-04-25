import cv2


def segment(img):
    """
    Segment the region of interest using Otsu's thresholding.

    Applies binary thresholding with Otsu's method to automatically
    determine the optimal threshold value, then extracts the ROI
    using the resulting mask.

    Args:
        img: Preprocessed grayscale image (numpy array).

    Returns:
        Tuple of (roi, mask) where roi is the segmented region
        and mask is the binary segmentation mask.
    """
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    roi = cv2.bitwise_and(img, img, mask=mask)
    return roi, mask
