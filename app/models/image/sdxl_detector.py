"""
SDXL AI Image Detector Model Wrapper
Handles loading the model and running inference
"""

import time
import logging
from PIL import Image
from typing import Dict, Any, Optional
import torch

from app.models.image.base import normalize_detector_result
from app.models.hf_loader import load_image_pipeline

# Set up logging
logger = logging.getLogger(__name__)

class SDXLDetector:
    """
    Wrapper for SDXL AI image detection model
    """

    name = "sdxl"
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the detector and load the model
        
        Args:
            device: Force specific device ('cpu', 'cuda', 'mps'). 
                   If None, auto-detect best available.
        """
        self.device = self._get_device(device)
        self.model = None
        self.model_name = "Organika/sdxl-detector"
        self.load_model()
    
    def _get_device(self, force_device: Optional[str] = None) -> str:
        """
        Determine the best available device for inference
        
        Args:
            force_device: Override auto-detection
            
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if force_device:
            return force_device
            
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self):
        """
        Load the SDXL detector model from the local Hugging Face cache.
        """
        try:
            logger.info(f"Loading SDXL detector on {self.device}...")
            start_time = time.time()
            
            self.model = load_image_pipeline(self.model_name, device=self.device)
            
            load_time = time.time() - start_time
            logger.info(f"✅ SDXL detector loaded in {load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to load SDXL detector: {e}")
            raise
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run detection on a single image
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Run inference
            start_time = time.time()
            results = self.model(image)
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Parse results with intelligent label detection
            ai_score, real_score = self._parse_results(results)
            
            out = normalize_detector_result(
                detector_name=self.name,
                model_used=self.model_name,
                ai_probability=float(ai_score),
                real_probability=float(real_score),
                confidence=float(max(ai_score, real_score)),
                inference_time_ms=inference_time,
                raw_results=results,
            )
            out["is_ai_generated"] = float(ai_score) > 0.5
            out["device_used"] = self.device
            return out
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _parse_results(self, results: list) -> tuple:
        """
        Parse model output to extract AI and real probabilities
        Handles different label formats intelligently
        
        Args:
            results: Raw model output list of dicts with 'label' and 'score'
            
        Returns:
            Tuple of (ai_probability, real_probability)
        """
        # Define keywords that indicate AI or real
        ai_keywords = ['ai', 'artificial', 'fake', 'generated', 'synthetic', 'LABEL_1']
        real_keywords = ['human', 'real', 'natural', 'authentic', 'original', 'LABEL_0']
        
        ai_score = 0.0
        real_score = 0.0
        
        for result in results:
            label = result['label'].lower()
            score = result['score']
            
            # Check for AI indicators
            if any(keyword in label for keyword in ai_keywords):
                ai_score = score
            # Check for real indicators
            elif any(keyword in label for keyword in real_keywords):
                real_score = score
        
        # If we couldn't identify by keywords, use position heuristic
        if ai_score == 0 and real_score == 0 and len(results) >= 2:
            ai_score = results[0]['score']
            real_score = results[1]['score']
        
        # Ensure both scores sum to approximately 1
        if ai_score > 0 and real_score == 0:
            real_score = 1.0 - ai_score
        elif real_score > 0 and ai_score == 0:
            ai_score = 1.0 - real_score
        
        return ai_score, real_score
    
    def unload_model(self):
        """Clean up model to free memory"""
        if self.model:
            del self.model
            self.model = None
            # Clear GPU cache if applicable
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Model unloaded")
