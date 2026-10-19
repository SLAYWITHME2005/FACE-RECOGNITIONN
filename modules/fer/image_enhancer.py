import cv2
import numpy as np
from typing import Dict, Tuple

class ImageEnhancer:
    """
    Adaptive Low-Light and Contrast Image Enhancer for DOSTH AI.
    Optimized for elderly indoor lighting, backlit rooms, and budget webcams.
    """
    def __init__(self, clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def auto_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Dynamically calculates and applies gamma correction based on image luminance.
        Brightens underexposed faces and normalizes harsh shadows.
        """
        if image is None or image.size == 0:
            return image
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean_val = float(np.mean(gray))
        
        # Avoid zero division
        if mean_val < 5.0:
            gamma = 0.35
        elif mean_val > 240.0:
            gamma = 1.4
        else:
            gamma = np.log(128.0 / 255.0) / np.log(mean_val / 255.0)
            gamma = float(np.clip(gamma, 0.35, 1.8))
            
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype="uint8")
        return cv2.LUT(image, table)

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L-channel (LAB color space)
        to improve local contrast without altering facial color tone.
        """
        if image is None or image.size == 0:
            return image
            
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            cl = self.clahe.apply(l_chan)
            merged = cv2.merge((cl, a_chan, b_chan))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        else:
            return self.clahe.apply(image)

    def denoise_bilateral(self, image: np.ndarray) -> np.ndarray:
        """
        Smooths webcam noise while preserving facial edges and expression wrinkles.
        """
        if image is None or image.size == 0:
            return image
        return cv2.bilateralFilter(image, d=5, sigmaColor=35, sigmaSpace=35)

    def compute_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """
        Computes lighting and clarity metrics:
        - brightness: Mean luminance (0-255)
        - contrast: Standard deviation of luminance
        - sharpness: Variance of Laplacian (blur metric)
        """
        if image is None or image.size == 0:
            return {'brightness': 0.0, 'contrast': 0.0, 'sharpness': 0.0, 'is_usable': False}
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        is_usable = (brightness >= 25.0 and sharpness >= 15.0)
        return {
            'brightness': round(brightness, 2),
            'contrast': round(contrast, 2),
            'sharpness': round(sharpness, 2),
            'is_usable': is_usable
        }

    def enhance(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Runs the full enhancement pipeline:
        1. Dynamic Gamma Correction
        2. Bilateral Denoising
        3. LAB CLAHE contrast boost
        Returns enhanced image and quality metrics.
        """
        if image is None or image.size == 0:
            return image, self.compute_quality_metrics(image)
            
        step1 = self.auto_gamma_correction(image)
        step2 = self.denoise_bilateral(step1)
        enhanced = self.apply_clahe(step2)
        metrics = self.compute_quality_metrics(enhanced)
        return enhanced, metrics
