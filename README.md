# Soccer Player Re-identification Tracker

A comprehensive system for tracking and re-identifying soccer players in video footage using deep learning techniques, computer vision, and multi-modal feature matching.

## Features

- **Multi-modal Player Tracking**: Combines deep learning embeddings, color histograms, and jersey number recognition
- **Robust Re-identification**: Handles player occlusions, temporary disappearances, and re-appearances
- **Real-time Processing**: Optimized for real-time or near real-time video processing
- **Configurable Parameters**: Easily adjustable tracking parameters and thresholds
- **Comprehensive Logging**: Track player statistics and performance metrics

## System Architecture

The system consists of several modular components:

1. **Object Detection**: YOLO-based player detection
2. **Feature Extraction**: Deep learning embeddings + color histograms + OCR
3. **Player Matching**: Multi-cue association using cosine similarity and IoU
4. **Tracking**: DeepSORT-based temporal tracking with fixed ID assignment
5. **Visualization**: Real-time display with bounding boxes and player information

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- OpenCV
- PyTorch

### Setup

1. Download the folder:
```
cd soccer-player-tracker
```

2. Create a virtual environment:
```bash
python -m venv venv
Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models:
   - **YOLO Detector**: Place your YOLO model (e.g., `yolo_detector.pt`) in the `models/` directory
   - **Re-ID Model**: Download OSNet model (`model.osnet.pth.tar-10`) and place in `models/` directory

## Usage

### Basic Usage

```bash
python main.py --input path/to/video.mp4 --output path/to/output.mp4
```

### Advanced Usage

```bash
python main.py \
    --input input_video.mp4 \
    --output tracked_output.mp4 \
    --detector models/custom_yolo.pt \
    --reid models/custom_reid.pth \
    --max-players 22 \
    --gallery-size 15 \
    --no-display
```

### Command Line Arguments

- `--input, -i`: Path to input video file (required)
- `--output, -o`: Path to output video file (optional)
- `--detector`: Path to YOLO detector model (default: `models/yolo_detector.pt`)
- `--reid`: Path to re-identification model (default: `models/model.osnet.pth.tar-10`)
- `--max-players`: Maximum number of players to track (default: 28)
- `--gallery-size`: Size of embedding gallery per player (default: 10)
- `--no-display`: Disable real-time display window

## Configuration

The system can be configured by modifying `config.py`:

### Key Configuration Options

```python
# Tracking parameters
MAX_ENTITIES = 28                    # Maximum players to track
EMBEDDING_GALLERY_SIZE = 10          # Embedding history per player
EMBEDDING_THRESHOLD = 0.8            # Similarity threshold for matching
INACTIVE_TIMEOUT = 10                # Frames before player becomes inactive

# DeepSORT parameters
MAX_AGE = 60                         # Maximum frames to keep lost tracks
N_INIT = 1                           # Frames to confirm a track
MAX_COSINE_DISTANCE = 0.4            # Maximum cosine distance for matching
```

## Architecture Details

### Core Components

#### 1. SoccerReIDTracker (`soccer_tracker.py`)
Main tracking class that orchestrates the entire pipeline:
- Manages player detection and tracking
- Handles ID assignment and re-assignment
- Coordinates feature extraction and matching

#### 2. PlayerFeatureExtractor (`feature_extractor.py`)
Extracts multiple types of features from player crops:
- **Deep Learning Embeddings**: Using OSNet or ViT models
- **Color Histograms**: HSV histograms from shirt regions
- **Jersey Numbers**: OCR-based number recognition

#### 3. PlayerMatcher (`player_matcher.py`)
Handles association of new detections with existing players:
- Jersey number matching (hard constraint)
- Embedding similarity matching
- Fallback IoU matching for occlusions

#### 4. Utility Functions (`utils.py`)
Common utility functions for:
- Bounding box operations
- Visualization and overlay drawing
- Video I/O operations

### Tracking Pipeline

1. **Detection**: YOLO detects players in each frame
2. **Feature Extraction**: (OSnet or ViT with torchreid)Extract embeddings, color histograms, and jersey numbers
3. **Temporal Tracking**: DeepSORT provides short-term tracking
4. **Re-identification**: Match detections across temporal gaps using features
5. **ID Management**: Assign and maintain consistent player IDs
6. **Visualization**: Draw bounding boxes and player information

## Model Requirements

### YOLO Detector
- Trained on soccer/football player detection

### Re-identification Model
- **OSNet**: Omni-Scale Network for person re-identification
- **Alternative**: Vision Transformer (ViT) models
- Pre-trained on person re-identification datasets

## Performance Optimization

### GPU Acceleration
- CUDA support for YOLO detection
- GPU-accelerated OCR with EasyOCR
- PyTorch GPU operations for embeddings

### Memory Management
- Limited embedding gallery per player
- Automatic cleanup of inactive players
- Efficient numpy operations

##  Challenges Faced

### 1. **Same ID for Multiple Players**
- Issue: Multiple players getting assigned the same ID when appearance is similar.
- Solutions tried:
  - Use multiple cues (embedding + color histogram + OCR).
  - Applied **temporal smoothing** via averaged embeddings but didn't work well.
  - Prevent same fixed_id assignment within the same frame using `seen_fixed_ids`.

### 2. **OCR Errors & Missing Jerseys**
- Issue: OCR often fails in poor lighting or blurry images.
- OCR is treated as an optional feature; fallback to embedding/histogram.
- Soft priority for matching using jersey number.

### 3. **Reappearance After Occlusion**
- Issue: Players often reappear after disappearing (e.g., leaving FOV).
- Solution applied:
  - Maintain an `inactive_ids` gallery.
  - Match reappearing players to this gallery using embeddings and IoU.

### 4. **Tracking Drift & Identity Switches**
- Issue: DeepSORT occasionally switches IDs.
- Solution applied:
  - Use fixed_id logic independent of `track_id`.
  - Assign fixed IDs manually and manage their lifecycle using `active_ids` and `track_to_fixed_id`.

### 5. **Embedding Drift**
- Issue: Embeddings can vary due to lighting or pose.
- Solution:
  - Temporal smoothing of embeddings via rolling average in gallery.
  - Only trust embedding if similarity exceeds a threshold.
  -But higher threshold causing non detection of players.

---
```
## Improvements and Experiences

- Integrate **Re-Ranking** (e.g., Euclidean + Jaccard) for better matching (though I have tried but not successfully differentiate between teammates).
- Use **MGAC**, **TransReID**, or **CTF strategies** for ReID.(experimented with Transformer based reid 'ViT_l_16' and also CTF for post processing thinking of giving large embeddings than OSNet but didn't performed better than osnet while being very slow on inference)
-Had to finetune OSNet model with the help of **Soccernet** dataset for reid (12GB) for providing better embeddings of soccer players.
---







