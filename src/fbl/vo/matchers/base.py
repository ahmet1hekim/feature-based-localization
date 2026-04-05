import numpy as np

class BaseMatcher:
    """
    Abstract Base Class for Visual Odometry matchers (e.g., SuperGlue, LightGlue, LoFTR).
    Encapsulates all neural-network specific preprocessing, extraction, and tensor inference.
    """
    def match(self, img0: np.ndarray, img1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Matches two images and returns valid coordinate pairs.
        
        Args:
            img0: BGR or Grayscale numpy array
            img1: BGR or Grayscale numpy array
            
        Returns:
            mkpts0: (N, 2) float32 numpy array of coordinates in img0
            mkpts1: (N, 2) float32 numpy array of coordinates in img1
            vis: BGR visualization image showing the matches
        """
        raise NotImplementedError("Must be implemented by subclasses")
