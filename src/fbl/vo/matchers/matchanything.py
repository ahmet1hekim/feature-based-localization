import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForKeypointMatching

from .base import BaseMatcher

class MatchAnythingMatcher(BaseMatcher):
    def __init__(self, conf_thresh=0.2):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf_thresh = conf_thresh
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

        print("[Matcher] Loading MatchAnything Processor...")
        local_dir = os.path.join(project_root, "src/externals/matchanything_eloftr_local")
        self.processor = AutoImageProcessor.from_pretrained(local_dir)
        print("[Matcher] Loading MatchAnything Model...")
        self.model = AutoModelForKeypointMatching.from_pretrained(local_dir).to(self.device).eval()
        print(f"[Matcher] MatchAnything Strategy ready on {self.device}")

    def match(self, img0: np.ndarray, img1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB) if img0.ndim == 3 else cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
        rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if img1.ndim == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        
        pil0 = Image.fromarray(rgb0)
        pil1 = Image.fromarray(rgb1)

        inputs = self.processor([pil0, pil1], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        image_sizes = [[(pil0.height, pil0.width), (pil1.height, pil1.width)]]
        matches = self.processor.post_process_keypoint_matching(
            outputs, image_sizes, threshold=self.conf_thresh
        )
        
        if not matches or len(matches) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), None
        match_data = matches[0]
        kpts0 = match_data.get("keypoints0", [])
        kpts1 = match_data.get("keypoints1", [])
        matches0 = match_data.get("matches0", [])    
                
        if isinstance(kpts0, torch.Tensor): kpts0 = kpts0.cpu().numpy()
        if isinstance(kpts1, torch.Tensor): kpts1 = kpts1.cpu().numpy()

        if "matches0" in match_data:
            matches0 = match_data["matches0"]
            if isinstance(matches0, torch.Tensor): matches0 = matches0.cpu().numpy()
            mkpts0, mkpts1 = [], []
            for i, m in enumerate(matches0):
                if m >= 0:
                    mkpts0.append(kpts0[i])
                    mkpts1.append(kpts1[m])
            mkpts0 = np.float32(mkpts0) if len(mkpts0) > 0 else np.zeros((0, 2), dtype=np.float32)
            mkpts1 = np.float32(mkpts1) if len(mkpts1) > 0 else np.zeros((0, 2), dtype=np.float32)
        else:
            mkpts0 = np.float32(kpts0) if len(kpts0) > 0 else np.zeros((0, 2), dtype=np.float32)
            mkpts1 = np.float32(kpts1) if len(kpts1) > 0 else np.zeros((0, 2), dtype=np.float32)

        vis = self._draw_matches(img0, img1, mkpts0, mkpts1)
        return mkpts0, mkpts1, vis

    def _draw_matches(self, img0: np.ndarray, img1: np.ndarray, mkpts0: np.ndarray, mkpts1: np.ndarray) -> np.ndarray:
        img0_vis = img0.copy() if img0.ndim == 3 else cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        img1_vis = img1.copy() if img1.ndim == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        h = max(img0_vis.shape[0], img1_vis.shape[0])
        w0 = img0_vis.shape[1]
        vis = np.zeros((h, img0_vis.shape[1] + img1_vis.shape[1], 3), dtype=np.uint8)
        vis[:img0_vis.shape[0], :img0_vis.shape[1]] = img0_vis
        vis[:img1_vis.shape[0], img0_vis.shape[1]:] = img1_vis
        
        for i in range(len(mkpts0)):
            pt0 = tuple(map(int, mkpts0[i]))
            pt1 = (int(mkpts1[i][0]) + w0, int(mkpts1[i][1]))
            cv2.circle(vis, pt0, 3, (0, 255, 0), -1)
            cv2.circle(vis, pt1,  3, (0, 255, 0), -1)
            cv2.line(vis, pt0, pt1, (0, 255, 0), 1)
        return vis
