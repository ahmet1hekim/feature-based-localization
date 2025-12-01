import os

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


def preprocess_image(img, center_x, center_y, target_w, target_h):
    h, w = img.shape[:2]

    half_w = target_w // 2
    half_h = target_h // 2

    # Calculate crop bounds clamped to image
    x1 = center_x - half_w
    y1 = center_y - half_h
    x2 = center_x + half_w
    y2 = center_y + half_h

    # Compute intersection with image bounds
    ix1 = max(x1, 0)
    iy1 = max(y1, 0)
    ix2 = min(x2, w)
    iy2 = min(y2, h)

    # Initialize black canvas of desired size
    cropped = np.zeros((target_h, target_w, 3), dtype=img.dtype)

    # Calculate where to place the valid image patch in the canvas
    start_x = ix1 - x1  # offset if crop window extends left beyond image
    start_y = iy1 - y1  # offset if crop window extends top beyond image

    # Width and height of the valid patch
    valid_w = ix2 - ix1
    valid_h = iy2 - iy1

    if valid_w > 0 and valid_h > 0:
        # Copy valid part from image to the black canvas
        cropped[start_y : start_y + valid_h, start_x : start_x + valid_w] = img[
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
    refined_x, refined_y = first_lock_x, first_lock_y

    while True:
        frame = recv.get_mat()
        if frame is None:
            continue

        h_frame, w_frame = frame.shape[:2]

        # Crop map around current refined pos
        map_proc_part = preprocess_image(
            map_img, int(refined_x), int(refined_y), w_frame, h_frame
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

        # Filter good matches by confidence and valid indices
        good_match_idxs = [i for i, m in enumerate(matches) if m >= 0 and conf[i] > 0.5]

        if len(good_match_idxs) > 5:  # only refine if enough matches
            # Calculate average displacement vector of matches from frame to map crop
            displacements = []
            for i in good_match_idxs:
                pt0 = kpts0[i]
                pt1 = kpts1[matches[i]]
                disp = pt1 - pt0  # displacement vector (map_kpt - frame_kpt)
                displacements.append(disp)

            mean_disp = np.mean(displacements, axis=0)

            # Update refined position by shifting with mean displacement
            refined_x += mean_disp[0]
            refined_y += mean_disp[1]

        print(
            f"Refined position: ({refined_x:.2f}, {refined_y:.2f}), Matches: {len(good_match_idxs)}"
        )

        # Visualization (optional)
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

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break


if __name__ == "__main__":
    main()
