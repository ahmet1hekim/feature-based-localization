import math
import os
from typing import Tuple

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
from externals.SuperGluePretrainedNetwork.models.matching import Matching
from externals.SuperGluePretrainedNetwork.models.utils import make_matching_plot
from helpers.receiver import Receiver

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
assets_dir = os.path.join(PROJECT_ROOT, "assets")

resize_float = True
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Matcher config ===
config = {
    "superpoint": {"nms_radius": 4, "keypoint_threshold": 0.005, "max_keypoints": 1024},
    "superglue": {
        "weights": "outdoor",
        "sinkhorn_iterations": 20,
        "match_threshold": 0.2,
    },
}
matching = Matching(config).eval().to(device)


def normalize_deg(a):
    return (a + 360.0) % 360.0


def angle_error_deg(pred, real):
    """
    Smallest signed difference between two angles in degrees
    Result in range [-180, 180]
    """
    diff = (pred - real + 180.0) % 360.0 - 180.0
    return diff


def preprocess_image(img, center_x, center_y, target_w, target_h, angle=0):
    h, w = img.shape[:2]

    if angle != 0:
        # Compute rotation matrix
        rot_mat = cv2.getRotationMatrix2D((center_x, center_y), -angle, 1.0)

        # Calculate the new bounding dimensions of the rotated image
        cos = abs(rot_mat[0, 0])
        sin = abs(rot_mat[0, 1])

        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Adjust rotation matrix to take into account translation
        rot_mat[0, 2] += (new_w / 2) - center_x
        rot_mat[1, 2] += (new_h / 2) - center_y

        # Perform the actual rotation with expanded size
        rotated = cv2.warpAffine(
            img,
            rot_mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Update center coordinates for cropping because of expanded image
        center_x = new_w // 2
        center_y = new_h // 2

        # Use expanded image dimensions
        h, w = rotated.shape[:2]

    else:
        rotated = img

    # Now crop the centered window from the rotated image
    half_w = target_w // 2
    half_h = target_h // 2

    x1 = center_x - half_w
    y1 = center_y - half_h
    x2 = center_x + half_w
    y2 = center_y + half_h

    # Clamp crop to rotated image bounds
    ix1 = max(x1, 0)
    iy1 = max(y1, 0)
    ix2 = min(x2, w)
    iy2 = min(y2, h)

    # Create black canvas
    cropped = np.zeros((target_h, target_w, 3), dtype=img.dtype)

    start_x = ix1 - x1
    start_y = iy1 - y1

    valid_w = ix2 - ix1
    valid_h = iy2 - iy1

    if valid_w > 0 and valid_h > 0:
        cropped[start_y : start_y + valid_h, start_x : start_x + valid_w] = rotated[
            iy1:iy2, ix1:ix2
        ]

    # Convert to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    return gray


def to_superpoint_tensor(gray_img, device):
    """
    gray_img: numpy array HxW (uint8)
    returns: torch tensor (1, 1, H, W) on the correct device
    """
    t = torch.from_numpy(gray_img).float() / 255.0
    t = t.unsqueeze(0).unsqueeze(0)
    return t.to(device)


def draw_superglue_matches(img0, img1, kpts0, kpts1, matches0, conf, conf_thresh=0.2):
    """
    img0, img1: grayscale or BGR (numpy)
    kpts0, kpts1: Nx2 keypoints
    matches0: array of size N0 with idx in kpts1 or -1
    conf: array of size N0 with confidence in [0,1]
    conf_thresh: minimum confidence to draw match
    """

    # Convert to BGR so matches are colored
    if len(img0.shape) == 2:
        img0_vis = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    else:
        img0_vis = img0.copy()

    if len(img1.shape) == 2:
        img1_vis = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_vis = img1.copy()

    # Concatenate images horizontally
    h = max(img0_vis.shape[0], img1_vis.shape[0])
    w0 = img0_vis.shape[1]
    vis = np.zeros((h, img0_vis.shape[1] + img1_vis.shape[1], 3), dtype=np.uint8)
    vis[: img0_vis.shape[0], : img0_vis.shape[1]] = img0_vis
    vis[: img1_vis.shape[0], img0_vis.shape[1] :] = img1_vis

    # Draw matches
    for i, j in enumerate(matches0):
        if j < 0:
            continue  # skip unmatched
        if conf[i] < conf_thresh:
            continue  # skip low-confidence matches

        pt0 = tuple(map(int, kpts0[i]))
        pt1 = tuple(map(int, kpts1[j]))
        pt1_shifted = (pt1[0] + w0, pt1[1])  # shift x for the right image

        color = (0, 255, 0)

        cv2.circle(vis, pt0, 3, color, -1)
        cv2.circle(vis, pt1_shifted, 3, color, -1)
        cv2.line(vis, pt0, pt1_shifted, color, 1)

    return vis


def count_good_matches(matches, conf, conf_thresh=0.2):
    matches = np.array(matches)
    conf = np.array(conf)
    valid_mask = (matches >= 0) & (conf >= conf_thresh)
    return np.sum(valid_mask)


# def main():
#     print("starting..")
#     recv = Receiver("127.0.0.1", 12345)
#     recv.start()

#     # --- Pose ---
#     locked_x, locked_y = 640.0, 360.0
#     locked_theta = 0.0  # predicted angle (deg)

#     # --- Error tracking ---
#     angle_err_sq_sum = 0.0
#     angle_err_count = 0

#     # --- Previous frame ---
#     past_frame = None

#     PIXEL_NOISE_THRESH = 0.9  # px
#     SCALE = 1.0  # world units per pixel

#     while True:
#         frame = recv.get_mat()
#         # real_angle = recv.get_float()
#         if frame is None:
#             continue

#         h, w = frame.shape[:2]
#         frame_proc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Initialization
#         if past_frame is None:
#             past_frame = frame_proc.copy()
#             continue

#         # SuperGlue matching
#         with torch.no_grad():
#             pred = matching(
#                 {
#                     "image0": to_superpoint_tensor(past_frame, device),
#                     "image1": to_superpoint_tensor(frame_proc, device),
#                 }
#             )
#             pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

#         matches = pred["matches0"]
#         kpts0 = pred["keypoints0"]
#         kpts1 = pred["keypoints1"]
#         conf = pred["matching_scores0"]

#         pts0, pts1 = [], []
#         for i, m in enumerate(matches):
#             if m >= 0 and conf[i] > 0.5:
#                 pts0.append(kpts0[i])
#                 pts1.append(kpts1[m])

#         used_matches = len(pts0)

#         if used_matches >= 8:
#             pts0 = np.float32(pts0)
#             pts1 = np.float32(pts1)

#             M, inliers = cv2.estimateAffinePartial2D(
#                 pts0,
#                 pts1,
#                 method=cv2.RANSAC,
#                 ransacReprojThreshold=3.0,
#                 maxIters=2000,
#                 confidence=0.99,
#             )

#             if M is not None and inliers is not None:
#                 if int(inliers.sum()) >= 6:
#                     cx, cy = w * 0.5, h * 0.5

#                     M3 = np.vstack([M, [0, 0, 1]])
#                     T_neg = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
#                     T_pos = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
#                     M_center = (T_pos @ M3 @ T_neg)[:2, :]

#                     tx_img = M_center[0, 2]
#                     ty_img = M_center[1, 2]

#                     rot_rad = math.atan2(M_center[1, 0], M_center[0, 0])
#                     rot_deg = math.degrees(rot_rad)

#                     # --- Update predicted angle ---
#                     locked_theta += rot_deg
#                     locked_theta = normalize_deg(locked_theta)

#                     if math.hypot(tx_img, ty_img) > PIXEL_NOISE_THRESH:
#                         th_rad = math.radians(locked_theta)
#                         dx = math.cos(th_rad) * tx_img - math.sin(th_rad) * ty_img
#                         dy = math.sin(th_rad) * tx_img + math.cos(th_rad) * ty_img
#                         locked_x += dx * SCALE
#                         locked_y -= dy * SCALE

#         past_frame = frame_proc.copy()

#         # --- Angle comparison ---
#         pred_angle = normalize_deg(-locked_theta)
#         # real_angle = real_angle

#         ang_err = angle_error_deg(pred_angle, 0)#real_angle)
#         angle_err_sq_sum += ang_err**2
#         angle_err_count += 1
#         rmse = math.sqrt(angle_err_sq_sum / angle_err_count)

#         print(
#             f"x={locked_x:.2f}, y={locked_y:.2f}, "
#             f"pred_th={pred_angle:.2f}°, real_th={real_angle:.2f}°, "
#             f"err={ang_err:+.2f}°, rmse={rmse:.2f}°, "
#             f"matches={used_matches}"
#         )


# def main():
#     print("starting..")
#     recv = Receiver("127.0.0.1", 12345)
#     recv.start()

#     locked_theta = 0.0  # predicted angle (deg, OpenCV frame)
#     past_frame = None

#     while True:
#         frame = recv.get_mat()
#         if frame is None:
#             continue

#         h, w = frame.shape[:2]
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Init
#         if past_frame is None:
#             past_frame = frame_gray.copy()
#             continue

#         # SuperGlue
#         with torch.no_grad():
#             pred = matching(
#                 {
#                     "image0": to_superpoint_tensor(past_frame, device),
#                     "image1": to_superpoint_tensor(frame_gray, device),
#                 }
#             )
#             pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

#         matches = pred["matches0"]
#         kpts0 = pred["keypoints0"]
#         kpts1 = pred["keypoints1"]
#         conf = pred["matching_scores0"]

#         pts0, pts1 = [], []
#         for i, m in enumerate(matches):
#             if m >= 0 and conf[i] > 0.5:
#                 pts0.append(kpts0[i])
#                 pts1.append(kpts1[m])

#         if len(pts0) >= 8:
#             pts0 = np.float32(pts0)
#             pts1 = np.float32(pts1)

#             M, inliers = cv2.estimateAffinePartial2D(
#                 pts0,
#                 pts1,
#                 method=cv2.RANSAC,
#                 ransacReprojThreshold=3.0,
#                 maxIters=2000,
#                 confidence=0.99,
#             )

#             if M is not None and inliers is not None and int(inliers.sum()) >= 6:
#                 # Recenter affine transform
#                 cx, cy = w * 0.5, h * 0.5
#                 M3 = np.vstack([M, [0, 0, 1]])

#                 T_neg = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])

#                 T_pos = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])

#                 M_center = (T_pos @ M3 @ T_neg)[:2, :]

#                 # Rotation extraction
#                 rot_rad = math.atan2(M_center[1, 0], M_center[0, 0])
#                 rot_deg = math.degrees(rot_rad)

#                 # Angle update
#                 locked_theta -= rot_deg
#                 locked_theta = normalize_deg(locked_theta)

#         # position update logic should go here
#         #

#         past_frame = frame_gray.copy()

#         print(f"predicted_angle = {locked_theta:.2f}°")


def main():
    print("starting..")
    recv = Receiver("127.0.0.1", 12345)
    recv.start()

    locked_theta = 0.0  # predicted angle (deg, OpenCV frame)
    locked_x = 640.0
    locked_y = 360.0

    past_frame_gray = None
    past_frame_color = None

    while True:
        frame = recv.get_mat()
        if frame is None:
            continue

        h, w = frame.shape[:2]

        frame_color = frame  # BGR (for preprocess_image)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Init
        if past_frame_gray is None:
            past_frame_gray = frame_gray.copy()
            past_frame_color = frame_color.copy()
            continue

        # =========================
        # 1) ROTATION ESTIMATION
        # =========================
        with torch.no_grad():
            pred = matching(
                {
                    "image0": to_superpoint_tensor(past_frame_gray, device),
                    "image1": to_superpoint_tensor(frame_gray, device),
                }
            )
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        matches = pred["matches0"]
        kpts0 = pred["keypoints0"]
        kpts1 = pred["keypoints1"]
        conf = pred["matching_scores0"]

        pts0, pts1 = [], []
        for i, m in enumerate(matches):
            if m >= 0 and conf[i] > 0.5:
                pts0.append(kpts0[i])
                pts1.append(kpts1[m])

        if len(pts0) >= 8:
            pts0 = np.float32(pts0)
            pts1 = np.float32(pts1)

            M, inliers = cv2.estimateAffinePartial2D(
                pts0,
                pts1,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=2000,
                confidence=0.99,
            )

            if M is not None and inliers is not None and int(inliers.sum()) >= 6:
                cx, cy = w * 0.5, h * 0.5
                M3 = np.vstack([M, [0, 0, 1]])

                T_neg = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])

                T_pos = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])

                M_center = (T_pos @ M3 @ T_neg)[:2, :]

                rot_rad = math.atan2(M_center[1, 0], M_center[0, 0])
                rot_deg = math.degrees(rot_rad)

                # Angle update (your convention)
                locked_theta -= rot_deg
                locked_theta = normalize_deg(locked_theta)

        # =========================
        # 2) TRANSLATION (CORRECT)
        # =========================
        cx = w // 2
        cy = h // 2

        # Rotate CURRENT frame into PAST frame orientation
        curr_aligned = preprocess_image(
            frame_color,
            cx,
            cy,
            w,
            h,
            angle=-rot_deg,  # RELATIVE rotation ONLY
        )

        prev_aligned = past_frame_gray  # NO rotation

        with torch.no_grad():
            pred_t = matching(
                {
                    "image0": to_superpoint_tensor(prev_aligned, device),
                    "image1": to_superpoint_tensor(curr_aligned, device),
                }
            )
            pred_t = {k: v[0].cpu().numpy() for k, v in pred_t.items()}

        matches_t = pred_t["matches0"]
        kpts0_t = pred_t["keypoints0"]
        kpts1_t = pred_t["keypoints1"]
        conf_t = pred_t["matching_scores0"]

        pts0_t, pts1_t = [], []
        for i, m in enumerate(matches_t):
            if m >= 0 and conf_t[i] > 0.8:
                pts0_t.append(kpts0_t[i])
                pts1_t.append(kpts1_t[m])

        if len(pts0_t) >= 8:
            pts0_t = np.float32(pts0_t)
            pts1_t = np.float32(pts1_t)

            v = pts1_t - pts0_t
            dx_img = np.median(v[:, 0])
            dy_img = np.median(v[:, 1])

            # accumulate in world frame
            th = math.radians(locked_theta)
            dx = math.cos(th) * dx_img - math.sin(th) * dy_img
            dy = math.sin(th) * dx_img + math.cos(th) * dy_img

            locked_x -= dx
            locked_y -= dy

        past_frame_gray = frame_gray.copy()
        past_frame_color = frame_color.copy()

        print(f"θ={locked_theta:.2f}°, x={locked_x:.2f}, y={locked_y:.2f}")


if __name__ == "__main__":
    main()
