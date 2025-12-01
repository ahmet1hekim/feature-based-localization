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


def main():
    print("starting..")
    recv = Receiver("127.0.0.1", 12345)
    recv.start()

    # Load the reference map image once
    map_path = os.path.join(assets_dir, "dag.jpg")
    map_img = cv2.imread(map_path)
    if map_img is None:
        print(f"Failed to load reference image at {map_path}")
        return

    cursor_x, cursor_y = 0, 0
    cursor_x_lim, cursor_y_lim = 0, 0

    h_map, w_map = map_img.shape[:2]
    max_matches_curr = 0
    first_lock_x, first_lock_y = 0, 0
    while True:
        frame = recv.get_mat()
        if frame is not None:
            # Get incoming frame size (width, height)
            h_frame, w_frame = frame.shape[:2]

            map_cursor_x, map_cursor_y = (
                w_frame * cursor_x + int(w_frame / 2),
                h_frame * cursor_y + int(h_frame / 2),
            )
            cursor_y_lim, cursor_x_lim = int(h_map / h_frame), int(w_map / w_frame)

            # Crop and preprocess map to match incoming frame size
            map_proc_part = preprocess_image(
                map_img, map_cursor_x, map_cursor_y, w_frame, h_frame
            )

            # Preprocess incoming frame (grayscale, tensor)
            frame_proc = preprocess_image(
                frame, int(w_frame / 2), int(h_frame / 2), w_frame, h_frame
            )

            with torch.no_grad():
                pred = matching(
                    {
                        "image0": to_superpoint_tensor(frame_proc, device),
                        "image1": to_superpoint_tensor(map_proc_part, device),
                    }
                )
                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

            matches = pred["matches0"]
            kpts0 = pred["keypoints0"]
            kpts1 = pred["keypoints1"]
            conf = pred["matching_scores0"]

            print(cursor_x, " ", cursor_y)
            print("\n\n")

            vis = draw_superglue_matches(
                frame_proc,
                map_proc_part,
                kpts0,
                kpts1,
                matches,
                conf,
                0.5,
            )
            curr_num_matches = count_good_matches(matches, conf, conf_thresh=0.5)
            if curr_num_matches > max_matches_curr:
                max_matches_curr = curr_num_matches
                first_lock_x, first_lock_y = map_cursor_x, map_cursor_y
            cv2.imshow("Matches", vis)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

            # move search pos across the map
            if cursor_x < cursor_x_lim - 1:
                cursor_x += 1
            else:
                if cursor_y < cursor_y_lim - 1:
                    cursor_x = 0
                    cursor_y += 1

                else:
                    cursor_x = 0
                    cursor_y = 0
                    if input() == "q":
                        max_matches_curr = 0
                        break

    print(
        "init lock at :\nmax_matches: ",
        max_matches_curr,
        " ",
        "pos:",
        first_lock_x,
        " ",
        first_lock_y,
        "\n",
    )

    cv2.destroyAllWindows()

    # Initialize refined pos at coarse search result
    locked_x, locked_y = first_lock_x, first_lock_y

    while True:
        frame = recv.get_mat()
        angle = recv.get_float()  # degrees, 0-360

        if frame is None:
            continue

        h_frame, w_frame = frame.shape[:2]

        map_proc_part = preprocess_image(
            map_img, int(locked_x), int(locked_y), w_frame, h_frame, -angle
        )

        frame_proc = preprocess_image(
            frame, int(w_frame / 2), int(h_frame / 2), w_frame, h_frame
        )

        with torch.no_grad():
            pred = matching(
                {
                    "image0": to_superpoint_tensor(frame_proc, device),
                    "image1": to_superpoint_tensor(map_proc_part, device),
                }
            )
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        matches = pred["matches0"]
        kpts0 = pred["keypoints0"]
        kpts1 = pred["keypoints1"]
        conf = pred["matching_scores0"]

        # filter by conf
        conf_filtered_ids = [
            i for i, m in enumerate(matches) if m >= 0 and conf[i] > 0.5
        ]

        # calculate mean difference from features
        if len(conf_filtered_ids) > 5:
            pos_diff = []
            for i in conf_filtered_ids:
                pt0 = kpts0[i]
                pt1 = kpts1[matches[i]]
                disp = pt1 - pt0
                pos_diff.append(disp)

            mean_diff = np.mean(pos_diff, axis=0)

            # Compensate translation update by rotating displacement vector back by angle
            # (Because map crop is rotated, pos_diff are relative to rotated map)

            angle_rad = math.radians(angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            # Rotate diff vector by angle to get it into map coordinates
            map_x_diff = mean_diff[0] * cos_a - mean_diff[1] * sin_a
            map_y_diff = mean_diff[0] * sin_a + mean_diff[1] * cos_a

            locked_x += map_x_diff
            locked_y += map_y_diff

        print(
            f"locked position: ({locked_x:.2f}, {locked_y:.2f}), Angle: {angle:.2f}, Matches: {len(conf_filtered_ids)}"
        )

        vis = draw_superglue_matches(
            frame_proc,
            map_proc_part,
            kpts0,
            kpts1,
            matches,
            conf,
            0.5,
        )
        cv2.imshow("Refined Matches", vis)

        if cv2.waitKey(1) & 0xFF == 27:
            break


if __name__ == "__main__":
    main()
