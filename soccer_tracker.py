"""
Main soccer player re-identification tracker
"""

import cv2
import numpy as np
import torch
from collections import deque
from typing import Dict, Any, Optional
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from config import TrackingConfig, VideoConfig, ModelPaths
from feature_extractor import PlayerFeatureExtractor
from player_matcher import PlayerMatcher
from utils import draw_tracking_overlay, setup_video_writer, validate_bbox


class SoccerReIDTracker:
    """
    Main tracker class for soccer player re-identification and tracking.
    """
    
    def __init__(self, 
                 detector_model_path: str,
                 reid_model_path: str,
                 max_entities: int = TrackingConfig.MAX_ENTITIES,
                 embedding_gallery_size: int = TrackingConfig.EMBEDDING_GALLERY_SIZE):
        """
        Initialize the soccer tracker.
        
        Args:
            detector_model_path: Path to YOLO detection model
            reid_model_path: Path to re-identification model
            max_entities: Maximum number of players to track
            embedding_gallery_size: Size of embedding gallery per player
        """
        # Initialize detector
        self.detector = YOLO(detector_model_path, verbose=False)
        if torch.cuda.is_available():
            self.detector.to('cuda')
        
        # Initialize feature extractor
        self.feature_extractor = PlayerFeatureExtractor(reid_model_path)
        
        # Initialize player matcher
        self.matcher = PlayerMatcher()
        
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(
            max_age=TrackingConfig.MAX_AGE,
            n_init=TrackingConfig.N_INIT,
            max_cosine_distance=TrackingConfig.MAX_COSINE_DISTANCE
        )
        
        # Player management
        self.active_ids = {}      # fixed_id → player metadata
        self.inactive_ids = {}    # fixed_id → player metadata
        self.available_ids = deque(range(max_entities))
        self.track_to_fixed_id = {}  # track_id (from DeepSORT) → fixed_id
        
        # Configuration
        self.embedding_gallery_size = embedding_gallery_size
        self.inactive_timeout = TrackingConfig.INACTIVE_TIMEOUT
    
    def update_player_gallery(self, player_meta: Dict[str, Any], embedding: torch.Tensor) -> None:
        """
        Update embedding gallery for a player.
        
        Args:
            player_meta: Player metadata dictionary
            embedding: New embedding to add
        """
        if 'embeddings' not in player_meta:
            player_meta['embeddings'] = []
        
        player_meta['embeddings'].append(embedding.cpu().numpy().astype(np.float32))
        
        # Keep only the most recent embeddings
        if len(player_meta['embeddings']) > self.embedding_gallery_size:
            player_meta['embeddings'] = player_meta['embeddings'][-self.embedding_gallery_size:]
    
    def process_existing_track(self, track, frame_idx: int, frame: np.ndarray) -> Optional[int]:
        """
        Process a track that already has a fixed ID assignment.
        
        Args:
            track: DeepSORT track object
            frame_idx: Current frame index
            frame: Current frame
        
        Returns:
            Fixed ID if successfully processed, None otherwise
        """
        track_id = track.track_id
        bbox = track.to_ltrb(orig=True, orig_strict=True)
        
        if not track.is_confirmed() or bbox is None or not validate_bbox(bbox):
            return None
        
        fixed_id = self.track_to_fixed_id.get(track_id)
        if fixed_id is None or fixed_id not in self.active_ids:
            return None
        
        # Update player metadata
        self.active_ids[fixed_id]['last_seen'] = frame_idx
        self.active_ids[fixed_id]['track_duration'] += 1
        self.active_ids[fixed_id]['last_bbox'] = bbox
        
        # Update embedding gallery
        embedding, histogram, jersey_number = self.feature_extractor.extract_features(frame, bbox)
        if embedding is not None:
            self.update_player_gallery(self.active_ids[fixed_id], embedding)
        
        # Update jersey number if detected
        if jersey_number is not None:
            self.active_ids[fixed_id]['jersey'] = jersey_number
        
        return fixed_id
    
    def process_new_track(self, track, frame_idx: int, frame: np.ndarray) -> Optional[int]:
        """
        Process a new track without fixed ID assignment.
        
        Args:
            track: DeepSORT track object
            frame_idx: Current frame index
            frame: Current frame
        
        Returns:
            Assigned fixed ID if successful, None otherwise
        """
        track_id = track.track_id
        bbox = track.to_ltrb(orig=True, orig_strict=True)
        
        if not track.is_confirmed() or bbox is None or not validate_bbox(bbox):
            return None
        
        # Extract features
        embedding, histogram, jersey_number = self.feature_extractor.extract_features(frame, bbox)
        if embedding is None:
            return None
        
        # Try to match with existing players
        matched_id = self.matcher.find_best_match(
            embedding, histogram, jersey_number, bbox, 
            self.active_ids, self.inactive_ids
        )
        
        if matched_id is not None:
            # Reactivate existing player
            if matched_id in self.inactive_ids:
                self.active_ids[matched_id] = self.inactive_ids.pop(matched_id)
            
            # Update player metadata
            self.active_ids[matched_id]['last_seen'] = frame_idx
            self.active_ids[matched_id]['track_duration'] += 1
            self.active_ids[matched_id]['last_bbox'] = bbox
            
            if jersey_number is not None:
                self.active_ids[matched_id]['jersey'] = jersey_number
            if histogram is not None:
                self.active_ids[matched_id]['hist'] = histogram
            
            self.update_player_gallery(self.active_ids[matched_id], embedding)
            self.track_to_fixed_id[track_id] = matched_id
            
            return matched_id
        
        elif self.available_ids:
            # Create new player
            fixed_id = self.available_ids.popleft()
            self.active_ids[fixed_id] = {
                'embeddings': [embedding.cpu().numpy().astype(np.float32)],
                'hist': histogram,
                'jersey': jersey_number,
                'last_seen': frame_idx,
                'track_duration': 1,
                'last_bbox': bbox
            }
            self.track_to_fixed_id[track_id] = fixed_id
            return fixed_id
        
        return None
    
    def cleanup_inactive_players(self, frame_idx: int, seen_ids: set) -> None:
        """
        Move players to inactive status if they haven't been seen recently.
        
        Args:
            frame_idx: Current frame index
            seen_ids: Set of player IDs seen in current frame
        """
        for fixed_id in list(self.active_ids.keys()):
            if fixed_id not in seen_ids:
                frames_since_seen = frame_idx - self.active_ids[fixed_id]['last_seen']
                if frames_since_seen > self.inactive_timeout:
                    # Move to inactive
                    self.inactive_ids[fixed_id] = self.active_ids.pop(fixed_id)
                    self.inactive_ids[fixed_id]['track_duration'] = 0
                    
                    # Clean up track mapping
                    track_ids_to_remove = [
                        tid for tid, fid in self.track_to_fixed_id.items() 
                        if fid == fixed_id
                    ]
                    for tid in track_ids_to_remove:
                        del self.track_to_fixed_id[tid]
    
    def run(self, video_path: str, output_path: Optional[str] = None) -> None:
        """
        Run the tracker on a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to output video (optional)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        writer = None
        frame_idx = 0
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or VideoConfig.DEFAULT_FPS
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Object detection
                results = self.detector(frame)
                detections = []
                
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        # Convert to DeepSORT format (x, y, w, h)
                        detections.append(([x1, y1, x2 - x1, y2 - y1], confidence))
                
                # Update tracker
                tracks = self.tracker.update_tracks(detections, frame=frame)
                seen_fixed_ids = set()
                
                # Process tracks
                for track in tracks:
                    track_id = track.track_id
                    
                    # Try processing as existing track first
                    if track_id in self.track_to_fixed_id:
                        fixed_id = self.process_existing_track(track, frame_idx, frame)
                        if fixed_id is not None and fixed_id not in seen_fixed_ids:
                            seen_fixed_ids.add(fixed_id)
                            self._draw_player_overlay(frame, track, fixed_id)
                    else:
                        # Process as new track
                        fixed_id = self.process_new_track(track, frame_idx, frame)
                        if fixed_id is not None and fixed_id not in seen_fixed_ids:
                            seen_fixed_ids.add(fixed_id)
                            self._draw_player_overlay(frame, track, fixed_id)
                
                # Cleanup inactive players
                self.cleanup_inactive_players(frame_idx, seen_fixed_ids)
                
                # Setup video writer if needed
                if output_path and writer is None:
                    frame_shape = (frame.shape[1], frame.shape[0])
                    writer = setup_video_writer(output_path, frame_shape, fps)
                
                # Write frame
                if writer:
                    writer.write(frame)
                
                # Display frame
                if VideoConfig.DISPLAY_WINDOW:
                    cv2.imshow('Soccer Player Tracker', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_idx += 1
                
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
    
    def _draw_player_overlay(self, frame: np.ndarray, track, fixed_id: int) -> None:
        """
        Draw tracking overlay for a player.
        
        Args:
            frame: Current frame
            track: DeepSORT track object
            fixed_id: Player's fixed ID
        """
        bbox = track.to_ltrb(orig=True, orig_strict=True)
        if bbox is None:
            return
        
        player_meta = self.active_ids.get(fixed_id, {})
        jersey_number = player_meta.get('jersey')
        track_duration = player_meta.get('track_duration', 0)
        
        colors = {
            'bbox': VideoConfig.BBOX_COLOR,
            'feet': VideoConfig.FEET_COLOR,
            'text': VideoConfig.TEXT_COLOR,
            'duration': VideoConfig.DURATION_COLOR
        }
        
        draw_tracking_overlay(
            frame, bbox, fixed_id, jersey_number, 
            track_duration, set(self.active_ids.keys()), colors
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current tracking statistics.
        
        Returns:
            Dictionary containing tracking statistics
        """
        return {
            'active_players': len(self.active_ids),
            'inactive_players': len(self.inactive_ids),
            'available_ids': len(self.available_ids),
            'active_player_ids': list(self.active_ids.keys()),
            'inactive_player_ids': list(self.inactive_ids.keys())
        }