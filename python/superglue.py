import os
import tempfile

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


def preprocess_image(img, target_size, resize_float=True):
    """
    Grayscale, resize/crop to target_size, convert to tensor and normalize.
    target_size = (width, height)
    """
    # Convert to grayscale if needed
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape[:2]
    tw, th = target_size

    # Crop image to target size if larger
    if w >= tw and h >= th:
        img_cropped = img[0:th, 0:tw]
    else:
        # If map is smaller than target, pad with zeros (black)
        img_cropped = np.zeros((th, tw), dtype=img.dtype)
        img_cropped[0:h, 0:w] = img

    if resize_float:
        img_cropped = img_cropped.astype(np.float32) / 255.0
    else:
        img_cropped = img_cropped.astype(np.float32)

    tensor = torch.from_numpy(img_cropped)[None, None].to(device)  # 1x1xHxW
    return img_cropped, tensor


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

    while True:
        frame = recv.get_mat()
        if frame is not None:
            # Get incoming frame size (width, height)
            h_frame, w_frame = frame.shape[:2]
            target_size = (w_frame, h_frame)

            # Crop and preprocess map to match incoming frame size
            map_cropped, inp0 = preprocess_image(map_img, target_size, resize_float)

            # Preprocess incoming frame (grayscale, tensor)
            frame_gray, inp1 = preprocess_image(frame, target_size, resize_float)

            with torch.no_grad():
                pred = matching({"image0": inp0, "image1": inp1})
                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

            matches = pred["matches0"]
            kpts0 = pred["keypoints0"]
            kpts1 = pred["keypoints1"]
            conf = pred["matching_scores0"]

            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid]
            color = cm.jet(mconf)[:, :3]  # float32 in [0,1]

            text = [
                "SuperGlue matches",
                f"#Keypoints0: {len(kpts0)}",
                f"#Keypoints1: {len(kpts1)}",
                f"#Matches: {len(mkpts0)}",
            ]
            vis_img = make_matching_plot(
                map_cropped,
                frame_gray,
                kpts0,
                kpts1,
                mkpts0,
                mkpts1,
                color,
                text,
                path=None,  # no saving
                show_keypoints=True,
                fast_viz=False,
                opencv_display=False,
                opencv_title="Matches",
            )

            if vis_img is not None:
                cv2.imshow("SuperGlue Matching", vis_img)
            else:
                cv2.imshow("SuperGlue Matching", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cv2.destroyAllWindows()
    os.remove(temp_vis_path)


if __name__ == "__main__":
    main()
