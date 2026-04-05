import cv2
import numpy as np
import torch

try:
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from externals.SuperGluePretrainedNetwork.models.matching import Matching
except ImportError:
    pass

from .base import BaseMatcher

class SuperGlueMatcher(BaseMatcher):
    def __init__(self, weights="outdoor", conf_thresh=0.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf_thresh = conf_thresh
        self.config = {
            "superpoint": {"nms_radius": 4, "keypoint_threshold": 0.005, "max_keypoints": 1024},
            "superglue": {
                "weights": weights,
                "sinkhorn_iterations": 20,
                "match_threshold": 0.2,
            },
        }
        print(f"[Matcher] Loading SuperGlue ({weights})...")
        self.matching = Matching(self.config).eval().to(self.device)
        print(f"[Matcher] Model ready on {self.device}")

    def _to_tensor(self, gray_img):
        t = torch.from_numpy(gray_img).float() / 255.0
        return t.unsqueeze(0).unsqueeze(0).to(self.device)

    def match(self, img0: np.ndarray, img1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) if img0.ndim == 3 else img0
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1

        with torch.no_grad():
            pred = self.matching({
                "image0": self._to_tensor(gray0),
                "image1": self._to_tensor(gray1),
            })
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        matches = pred["matches0"]
        kpts0   = pred["keypoints0"]
        kpts1   = pred["keypoints1"]
        conf    = pred["matching_scores0"]

        mkpts0, mkpts1 = [], []
        for i, m in enumerate(matches):
            if m >= 0 and conf[i] > self.conf_thresh:
                mkpts0.append(kpts0[i])
                mkpts1.append(kpts1[m])

        mkpts0 = np.float32(mkpts0) if len(mkpts0) > 0 else np.zeros((0, 2), dtype=np.float32)
        mkpts1 = np.float32(mkpts1) if len(mkpts1) > 0 else np.zeros((0, 2), dtype=np.float32)

        # Visualization
        vis = self._draw_matches(gray0, gray1, kpts0, kpts1, matches, conf)
        return mkpts0, mkpts1, vis

    def _draw_matches(self, img0, img1, kpts0, kpts1, matches0, conf):
        img0_vis = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        img1_vis = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        h = max(img0_vis.shape[0], img1_vis.shape[0])
        w0 = img0_vis.shape[1]
        vis = np.zeros((h, img0_vis.shape[1] + img1_vis.shape[1], 3), dtype=np.uint8)
        vis[:img0_vis.shape[0], :img0_vis.shape[1]] = img0_vis
        vis[:img1_vis.shape[0], img0_vis.shape[1]:]  = img1_vis
        
        for i, j in enumerate(matches0):
            if j < 0 or conf[i] < self.conf_thresh:
                continue
            pt0 = tuple(map(int, kpts0[i]))
            pt1 = (int(kpts1[j][0]) + w0, int(kpts1[j][1]))
            cv2.circle(vis, pt0, 3, (0, 255, 0), -1)
            cv2.circle(vis, pt1,  3, (0, 255, 0), -1)
            cv2.line(vis, pt0, pt1, (0, 255, 0), 1)
        return vis
