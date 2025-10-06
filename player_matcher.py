"""
Player matching module for associating detections with existing players
"""

import numpy as np
import torch
from typing import Optional, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

from config import TrackingConfig
from utils import iou


class PlayerMatcher:
    """
    Handles matching of new detections to existing players using multiple cues.
    """
    
    def __init__(self, 
                 emb_threshold: float = TrackingConfig.EMBEDDING_THRESHOLD,
                 hist_threshold: float = TrackingConfig.HISTOGRAM_THRESHOLD,
                 iou_threshold: float = TrackingConfig.IOU_THRESHOLD):
        """
        Initialize the player matcher.
        
        Args:
            emb_threshold: Minimum embedding similarity threshold
            hist_threshold: Minimum histogram similarity threshold  
            iou_threshold: Minimum IoU threshold for fallback matching
        """
        self.emb_threshold = emb_threshold
        self.hist_threshold = hist_threshold
        self.iou_threshold = iou_threshold
    
    def match_to_gallery(self, 
                        embedding: torch.Tensor, 
                        histogram: np.ndarray, 
                        jersey_number: Optional[int], 
                        gallery: Dict[int, Dict[str, Any]], 
                        jersey_first: bool = True) -> Optional[int]:
        """
        Match features to existing player gallery.
        
        Args:
            embedding: Player embedding tensor
            histogram: Color histogram
            jersey_number: Jersey number (if detected)
            gallery: Gallery of existing players
            jersey_first: Whether to prioritize jersey number matching
        
        Returns:
            Matched player ID or None
        """
        if not gallery:
            return None
        
        embedding_np = embedding.cpu().numpy().astype(np.float32)
        best_match = None
        best_score = 0.0
        
        for player_id, meta in gallery.items():
            if not meta.get('embeddings'):
                continue
            
            # Jersey number as hard constraint (if jersey_first is True)
            if jersey_first and jersey_number is not None and meta.get('jersey') is not None:
                if jersey_number == meta['jersey']:
                    return player_id
            
            try:
                # Calculate embedding similarity
                embedding_similarities = [
                    cosine_similarity([embedding_np], [emb])[0][0] 
                    for emb in meta['embeddings']
                ]
                embedding_similarity = max(embedding_similarities)
                
                # Calculate histogram similarity (if available)
                # histogram_similarity = cosine_similarity([histogram], [meta['hist']])[0][0]
                
                # Combined score (currently using only embedding)
                # score=0.7 * embedding_similarity + 0.3 * histogram_similarity
                score = embedding_similarity  
                
                if score > best_score and embedding_similarity > self.emb_threshold:
                    best_score = score
                    best_match = player_id
                    
            except Exception as e:
                print(f"Warning: Error calculating similarity for player {player_id}: {e}")
                continue
        
        return best_match
    
    def match_by_iou(self, 
                    current_bbox: list, 
                    gallery: Dict[int, Dict[str, Any]]) -> Optional[int]:
        """
        Fallback matching using IoU with last known bounding boxes.
        
        Args:
            current_bbox: Current bounding box [x1, y1, x2, y2]
            gallery: Gallery of existing players
        
        Returns:
            Matched player ID or None
        """
        for player_id, meta in gallery.items():
            if 'last_bbox' in meta:
                iou_score = iou(current_bbox, meta['last_bbox'])
                if iou_score > self.iou_threshold:
                    return player_id
        
        return None
    
    def find_best_match(self, 
                       embedding: torch.Tensor, 
                       histogram: np.ndarray, 
                       jersey_number: Optional[int], 
                       bbox: list,
                       active_gallery: Dict[int, Dict[str, Any]], 
                       inactive_gallery: Dict[int, Dict[str, Any]]) -> Optional[int]:
        """
        Find the best match across both active and inactive galleries.
        
        Args:
            embedding: Player embedding tensor
            histogram: Color histogram  
            jersey_number: Jersey number (if detected)
            bbox: Current bounding box
            active_gallery: Gallery of active players
            inactive_gallery: Gallery of inactive players
        
        Returns:
            Best matched player ID or None
        """
        # First try active gallery (jersey first, then embedding)
        matched_id = self.match_to_gallery(
            embedding, histogram, jersey_number, active_gallery, jersey_first=True
        )
        
        if matched_id is not None:
            return matched_id
        
        # Then try inactive gallery (for re-appearing players)
        matched_id = self.match_to_gallery(
            embedding, histogram, jersey_number, inactive_gallery, jersey_first=True
        )
        
        if matched_id is not None:
            return matched_id
        
        # Fallback: IoU matching with inactive players
        matched_id = self.match_by_iou(bbox, inactive_gallery)
        
        return matched_id