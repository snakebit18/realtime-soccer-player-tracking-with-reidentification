"""
Feature extraction module for player re-identification
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
import easyocr
from torchreid.reid.utils import FeatureExtractor
import torchvision.models as models
import torchvision.transforms as transforms

from config import FeatureConfig
from utils import extract_jersey_number, crop_shirt_region, calculate_color_histogram


class PlayerFeatureExtractor:
    """
    Feature extractor for player re-identification using deep learning embeddings,
    color histograms, and OCR for jersey numbers.
    """
    
    def __init__(self, reid_model_path: str):
        """
        Initialize the feature extractor.
        
        Args:
            reid_model_path: Path to the re-identification model
        """
        self.device = FeatureConfig.DEVICE if torch.cuda.is_available() else 'cpu'
        
        # Initialize OCR reader
        self.ocr = easyocr.Reader(
            FeatureConfig.OCR_LANGUAGES, 
            gpu=FeatureConfig.USE_GPU_OCR and torch.cuda.is_available()
        )
        
        # Initialize re-identification model
        self.extractor = FeatureExtractor(
            model_name=FeatureConfig.REID_MODEL_NAME,
            model_path=reid_model_path,
            device=self.device
        )
        
        # Alternative ViT model initialization (currently commented out)
        # self._init_vit_model()
    
    def _init_vit_model(self):
        """
        Initialize Vision Transformer model (alternative to OSNet).
        Currently commented out in the original code.
        """
        self.extractor = models.vit_l_16(weights=None)
        checkpoint = torch.load('path/to/vit/model.pth')
        if 'state_dict' in checkpoint:
            state_dict = {k.replace('state_dict.', ''): v for k, v in checkpoint['state_dict'].items()}
            try:
                self.extractor.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"Warning: Error loading state dict: {e}")
        else:
            print("Warning: Unexpected state dict format")
        
        self.extractor = self.extractor.to(self.device)
        self.extractor.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    
    def extract_features(self, frame: np.ndarray, bbox: List[int]) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[int]]:
        """
        Extract features from a player crop including embedding, color histogram, and jersey number.
        
        Args:
            frame: Input frame
            bbox: Bounding box coordinates [x1, y1, x2, y2]
        
        Returns:
            Tuple of (embedding, histogram, jersey_number)
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Validate crop
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
            return None, None, None
        
        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None, None, None
        
        # Extract embedding
        embedding = self._extract_embedding(crop)
        
        # Extract color histogram from shirt region
        histogram = self._extract_color_histogram(frame, bbox)
        
        # Extract jersey number
        jersey_number = self._extract_jersey_number(crop)
        
        return embedding, histogram, jersey_number
    
    def _extract_embedding(self, crop: np.ndarray) -> Optional[torch.Tensor]:
        """
        Extract deep learning embedding from player crop.
        
        Args:
            crop: Cropped player image
        
        Returns:
            Normalized embedding tensor
        """
        try:
            # Using OSNet extractor
            embedding = self.extractor([crop])[0]
            embedding = F.normalize(embedding, p=2, dim=0)
            return embedding
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
        
        # Alternative ViT extraction (commented out)
        # try:
        #     img = self.transform(crop).unsqueeze(0).to(self.device)
        #     with torch.no_grad():
        #         embedding = self.extractor(img)
        #         embedding = F.normalize(embedding, p=2, dim=1)[0]
        #     return embedding
        # except Exception as e:
        #     print(f"Error extracting ViT embedding: {e}")
        #     return None
    
    def _extract_color_histogram(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """
        Extract color histogram from shirt region.
        
        Args:
            frame: Input frame
            bbox: Bounding box coordinates
        
        Returns:
            Flattened color histogram
        """
        try:
            # Crop shirt region (top 40% of bounding box)
            shirt_crop = crop_shirt_region(frame, bbox, FeatureConfig.SHIRT_CROP_RATIO)
            
            # Calculate color histogram
            histogram = calculate_color_histogram(
                shirt_crop, 
                FeatureConfig.HIST_H_BINS, 
                FeatureConfig.HIST_S_BINS
            )
            
            return histogram
        except Exception as e:
            print(f"Error extracting color histogram: {e}")
            return None
    
    def _extract_jersey_number(self, crop: np.ndarray) -> Optional[int]:
        """
        Extract jersey number using OCR.
        
        Args:
            crop: Cropped player image
        
        Returns:
            Jersey number as integer or None
        """
        try:
            ocr_results = self.ocr.readtext(crop)
            return extract_jersey_number(ocr_results)
        except Exception as e:
            print(f"Error extracting jersey number: {e}")
            return None