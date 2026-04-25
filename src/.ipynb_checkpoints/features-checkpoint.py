import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def extract_texture(roi):
    """
    Extract GLCM texture features from the ROI.

    Computes contrast, energy, homogeneity, and correlation
    from the Gray-Level Co-occurrence Matrix.

    Args:
        roi: Segmented region of interest (grayscale numpy array).

    Returns:
        List of [contrast, energy, homogeneity, correlation].
    """
    glcm = graycomatrix(roi, distances=[1], angles=[0], levels=256,
                         symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return [contrast, energy, homogeneity, correlation]


def extract_shape(mask):
    """
    Extract contour-based shape features from the segmentation mask.

    Finds the largest contour and computes area, perimeter,
    and circularity.

    Args:
        mask: Binary segmentation mask (numpy array).

    Returns:
        List of [area, perimeter, circularity].
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0, 0, 0]
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
    return [area, perimeter, circularity]


def extract_features(roi, mask):
    """
    Extract combined texture and shape features from an image.

    Concatenates GLCM texture features and contour-based shape
    features into a single feature vector.

    Args:
        roi: Segmented region of interest (grayscale numpy array).
        mask: Binary segmentation mask (numpy array).

    Returns:
        List of 7 features: [contrast, energy, homogeneity,
        correlation, area, perimeter, circularity].
    """
    return extract_texture(roi) + extract_shape(mask)
