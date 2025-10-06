"""
Configuration file for Soccer Player Re-identification Tracker
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"

# Ensure directories exist
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Model paths
class ModelPaths:
    DETECTOR_MODEL = MODEL_DIR / "yolo_detector.pt"
    REID_MODEL = MODEL_DIR / "model.osnet.pth.tar-10"
    
    # Alternative ViT model path (currently commented out in main code)
    VIT_MODEL = MODEL_DIR / "model_soup.deit_l.pth"

# Tracking parameters, finetune the parameters here for better results
class TrackingConfig:
    MAX_ENTITIES = 28
    EMBEDDING_GALLERY_SIZE = 10
    TEMPORAL_WINDOW = 5
    
    # DeepSORT parameters
    MAX_AGE = 20
    N_INIT = 1
    MAX_COSINE_DISTANCE = 0.4
    
    # Matching thresholds
    EMBEDDING_THRESHOLD = 0.8
    HISTOGRAM_THRESHOLD = 0.6
    IOU_THRESHOLD = 0.5
    
    # Timeout settings
    INACTIVE_TIMEOUT = 10  # frames before moving to inactive

# Feature extraction settings
class FeatureConfig:
    DEVICE = 'cuda'
    REID_MODEL_NAME = 'osnet_x1_0'
    OCR_LANGUAGES = ['en']
    USE_GPU_OCR = True
    
    # Histogram parameters
    HIST_H_BINS = 50
    HIST_S_BINS = 60
    SHIRT_CROP_RATIO = 0.4  # Top 40% of bounding box for shirt color

# Video processing settings
class VideoConfig:
    DEFAULT_FPS = 30
    FOURCC = 'mp4v'
    DISPLAY_WINDOW = True
    
    # Visualization settings
    BBOX_COLOR = (0, 0, 255)  # Red
    FEET_COLOR = (0, 255, 0)  # Green
    TEXT_COLOR = (255, 255, 255)  # White
    DURATION_COLOR = (255, 0, 0)  # Blue
    
    BBOX_THICKNESS = 2
    TEXT_THICKNESS = 2
    FONT_SCALE = 0.6
    
    # Feet ellipse parameters
    FEET_ELLIPSE_WIDTH = 30
    FEET_ELLIPSE_HEIGHT = 20
    FEET_ELLIPSE_ANGLE = 0
    FEET_ELLIPSE_START_ANGLE = -40
    FEET_ELLIPSE_END_ANGLE = 220

# Environment settings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"