"""
Utility functions for the Soccer Player Re-identification Tracker
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional

def iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Compute intersection over union for fallback association.
    
    Args:
        boxA: Bounding box coordinates [x1, y1, x2, y2]
        boxB: Bounding box coordinates [x1, y1, x2, y2]
    
    Returns:
        IoU value between 0 and 1
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def extract_jersey_number(ocr_results: List) -> Optional[int]:
    """
    Extract jersey number from OCR results.
    
    Args:
        ocr_results: Results from EasyOCR
    
    Returns:
        Jersey number as integer or None if not found
    """
    if not ocr_results:
        return None
    
    texts = [result[1] for result in ocr_results if any(c.isdigit() for c in result[1])]
    if not texts:
        return None
    
    try:
        jersey_number = int(''.join(filter(str.isdigit, texts[0])))
        return jersey_number
    except (ValueError, IndexError):
        return None

def crop_shirt_region(frame: np.ndarray, bbox: List[int], crop_ratio: float = 0.4) -> np.ndarray:
    """
    Crop the shirt region from a player's bounding box.
    
    Args:
        frame: Input frame
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        crop_ratio: Ratio of the bounding box height to use for shirt region
    
    Returns:
        Cropped shirt region
    """
    x1, y1, x2, y2 = map(int, bbox)
    height = y2 - y1
    shirt_y2 = y1 + int(crop_ratio * height)
    return frame[y1:shirt_y2, x1:x2]

def calculate_color_histogram(image: np.ndarray, h_bins: int = 50, s_bins: int = 60) -> np.ndarray:
    """
    Calculate color histogram for an image region.
    
    Args:
        image: Input image in BGR format
        h_bins: Number of hue bins
        s_bins: Number of saturation bins
    
    Returns:
        Normalized histogram
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def draw_tracking_overlay(frame: np.ndarray, 
                         bbox: List[int], 
                         fixed_id: int, 
                         jersey_number: Optional[int] = None,
                         track_duration: int = 0,
                         active_ids: set = None,
                         colors: dict = None) -> None:
    """
    Draw tracking overlay on the frame.
    
    Args:
        frame: Input frame to draw on
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        fixed_id: Player's fixed ID
        jersey_number: Jersey number (optional)
        track_duration: Duration of tracking
        active_ids: Set of currently active IDs
        colors: Color configuration dictionary
    """
    if colors is None:
        colors = {
            'bbox': (0, 0, 255),
            'feet': (0, 255, 0),
            'text': (255, 255, 255),
            'duration': (255, 0, 0)
        }
    
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), colors['bbox'], 2)
    
    # Draw feet ellipse
    feet_x, feet_y = (x1 + x2) // 2, y2 - 10
    cv2.ellipse(frame, (feet_x, feet_y), (30, 20), 0, -40, 220, colors['feet'], 2)
    
    # Create label
    label = f"ID {fixed_id}"
    if jersey_number is not None:
        label += f" #{jersey_number}"
    
    # Draw text
    cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text'], 2)
    cv2.putText(frame, f'D: {track_duration}', (x1, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['duration'], 2)
    
    # Draw active IDs (optional)
    # if active_ids:
    #     cv2.putText(frame, f"Active: {list(active_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text'], 2)

def validate_bbox(bbox: List[float]) -> bool:
    """
    Validate bounding box coordinates.
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
    
    Returns:
        True if valid, False otherwise
    """
    if len(bbox) != 4:
        return False
    
    x1, y1, x2, y2 = bbox
    return x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0

def setup_video_writer(output_path: str, frame_shape: Tuple[int, int], fps: int = 30) -> cv2.VideoWriter:
    """
    Setup video writer for output.
    
    Args:
        output_path: Path to output video
        frame_shape: Shape of frames (width, height)
        fps: Frames per second
    
    Returns:
        Configured VideoWriter object
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, frame_shape)