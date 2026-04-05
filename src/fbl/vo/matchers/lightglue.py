import cv2
import numpy as np
import torch
from .base import BaseMatcher

import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path: sys.path.insert(0, project_root)
from externals.LightGlue.lightglue import LightGlue, SuperPoint
from externals.LightGlue.lightglue.utils import rbd

class LightGlueMatcher(BaseMatcher):
    def __init__(self, extractor_type="superpoint", max_keypoints=1024, conf_thresh=0.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf_thresh = conf_thresh
        
        print(f"[Matcher] Loading {extractor_type.capitalize()} Extractor...")
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
        print("[Matcher] Loading LightGlue...")
        self.matcher = LightGlue(features=extractor_type).eval().to(self.device)
        print(f"[Matcher] LightGlue Strategy ready on {self.device}")

    def _to_tensor(self, bgr_img: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB) if bgr_img.ndim == 3 else bgr_img
        image = rgb.transpose((2, 0, 1)) if rgb.ndim == 3 else rgb[None]
        t = torch.tensor(image / 255.0, dtype=torch.float).to(self.device)
        return t.unsqueeze(0)

    def match(self, img0: np.ndarray, img1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t0 = self._to_tensor(img0)
        t1 = self._to_tensor(img1)

        with torch.no_grad():
            feats0 = self.extractor.extract(t0)
            feats1 = self.extractor.extract(t1)
            pred = self.matcher({"image0": feats0, "image1": feats1})
            
            f0, f1, m01 = rbd(feats0), rbd(feats1), rbd(pred)

        kpts0, kpts1, idxs = f0["keypoints"], f1["keypoints"], m01["matches"]
        
        valid_idx = idxs[:, 0] != -1
        idxs = idxs[valid_idx]

        m_kpts0 = kpts0[idxs[:, 0]].cpu().numpy()
        m_kpts1 = kpts1[idxs[:, 1]].cpu().numpy()
        conf = m01["scores"][valid_idx].cpu().numpy()

        valid = conf > self.conf_thresh
        mkpts0 = m_kpts0[valid]
        mkpts1 = m_kpts1[valid]
        valid_conf = conf[valid]
        
        mkpts0 = np.float32(mkpts0) if len(mkpts0) > 0 else np.zeros((0, 2), dtype=np.float32)
        mkpts1 = np.float32(mkpts1) if len(mkpts1) > 0 else np.zeros((0, 2), dtype=np.float32)

        vis = self._draw_matches(img0, img1, mkpts0, mkpts1, valid_conf)
        return mkpts0, mkpts1, vis

    def _draw_matches(self, img0: np.ndarray, img1: np.ndarray, mkpts0: np.ndarray, mkpts1: np.ndarray, conf: np.ndarray) -> np.ndarray:
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
