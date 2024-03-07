
import numpy as np
def bbox_to_corners(bbox):
    """
    Convert YOLOv5 bounding box to four corner points.

    Args:
        bbox (list): List containing four values representing the bounding box.

    Returns:
        list: List of four tuples representing the four corner points of the bounding box.
    """
    x1, y1, x2, y2 = bbox
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]  # Clockwise starting from top-left
    return corners

# Example usage
bbox = [100, 200, 300, 400]  # Example YOLOv5 bounding box [x1, y1, x2, y2]
corners = bbox_to_corners(bbox)
print(corners)
print(np.array(corners))


