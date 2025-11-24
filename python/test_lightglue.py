import os
from collections import defaultdict

import cv2
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_dir = "/home/alex/Work/bitirme/images"
output_dir = os.path.join(os.getcwd(), "assets", "matched_outputs", "lightglue")
os.makedirs(output_dir, exist_ok=True)

# === Initialize extractor and matcher ===
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)


# === Visualization function ===
def draw_matches(img0, img1, kpts0, kpts1, confidences=None, out_path=None):
    # Convert to numpy arrays
    img0_np = (img0.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img1_np = (img1.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    if img0_np.ndim == 2 or img0_np.shape[2] == 1:
        img0_np = cv2.cvtColor(img0_np, cv2.COLOR_GRAY2RGB)
    if img1_np.ndim == 2 or img1_np.shape[2] == 1:
        img1_np = cv2.cvtColor(img1_np, cv2.COLOR_GRAY2RGB)

    # Prepare canvas
    H = max(img0_np.shape[0], img1_np.shape[0])
    W = img0_np.shape[1] + img1_np.shape[1]
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[: img0_np.shape[0], : img0_np.shape[1]] = img0_np
    canvas[: img1_np.shape[0], img0_np.shape[1] :] = img1_np

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(canvas)
    ax.axis("off")

    # Normalize confidences for colormap
    if confidences is None:
        confidences = np.ones(len(kpts0))
    confidences = np.clip(confidences, 0, 1)
    colors = cm.jet(confidences)

    # Draw matches
    offset = img0_np.shape[1]
    for pt0, pt1, color in zip(kpts0.cpu().numpy(), kpts1.cpu().numpy(), colors):
        x0, y0 = pt0
        x1, y1 = pt1
        x1 += offset
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=1)
        ax.add_patch(patches.Circle((x0, y0), 2.5, color=color))
        ax.add_patch(patches.Circle((x1, y1), 2.5, color=color))

    # Add overlay text
    ax.text(
        10,
        20,
        "LightGlue Matching",
        color="white",
        fontsize=14,
        weight="bold",
        backgroundcolor="black",
    )
    ax.text(
        10,
        45,
        f"Matches: {len(kpts0)}",
        color="white",
        fontsize=12,
        backgroundcolor="black",
    )

    # Save result
    if out_path:
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=200)
        plt.close(fig)
    else:
        plt.show()


# === Main processing ===
def find_matching_pairs(image_dir):
    """Find all jpg files and their corresponding folders"""
    pairs = []

    # Get all jpg files in the directory
    jpg_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]

    for jpg_file in jpg_files:
        base_name = os.path.splitext(jpg_file)[0]  # Remove .jpg extension
        folder_path = os.path.join(image_dir, base_name)

        # Check if corresponding folder exists
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # Get all image files in the folder
            folder_images = []
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                folder_images.extend(
                    [f for f in os.listdir(folder_path) if f.lower().endswith(ext)]
                )

            # Create pairs: (main_image, folder_image)
            for folder_img in folder_images:
                pairs.append((jpg_file, os.path.join(base_name, folder_img)))

    return pairs


# Find all matching pairs
matching_pairs = find_matching_pairs(image_dir)
print(f"Found {len(matching_pairs)} image pairs to process.")

if not matching_pairs:
    print("No matching pairs found. Please check your directory structure.")
    print("Expected structure:")
    print("image_dir/")
    print("├── x.jpg")
    print("├── x/")
    print("│   ├── image1.jpg")
    print("│   ├── image2.jpg")
    print("│   └── ...")
    print("├── y.jpg")
    print("├── y/")
    print("│   ├── image1.jpg")
    print("│   └── ...")

# Process each pair
for main_img, folder_img in matching_pairs:
    main_img_path = os.path.join(image_dir, main_img)
    folder_img_path = os.path.join(image_dir, folder_img)

    print(f"Processing: {main_img} + {folder_img}")

    try:
        # Load images
        img1 = load_image(main_img_path).to(device)
        img2 = load_image(folder_img_path).to(device)

        # Extract features
        feats1 = extractor.extract(img1)
        feats2 = extractor.extract(img2)

        # Match features
        matches = matcher({"image0": feats1, "image1": feats2})
        feats1, feats2, matches = [rbd(x) for x in [feats1, feats2, matches]]

        idxs = matches["matches"]
        valid = idxs[:, 0] != -1
        idxs = idxs[valid]

        if idxs.shape[0] == 0:
            print(f"No matches found for {main_img} vs {folder_img}")
            continue

        kpts1 = feats1["keypoints"][idxs[:, 0]]
        kpts2 = feats2["keypoints"][idxs[:, 1]]

        # Create output filename
        main_base = os.path.splitext(main_img)[0]
        folder_img_name = os.path.splitext(os.path.basename(folder_img))[0]
        out_filename = f"{main_base}_vs_{folder_img_name}_lightglue.png"
        out_path = os.path.join(output_dir, out_filename)

        # Draw and save matches
        conf = matches.get("scores")
        confidences = conf[valid].detach().cpu().numpy() if conf is not None else None

        draw_matches(img1, img2, kpts1, kpts2, confidences, out_path)
        print(f"Saved: {out_path} (Matches: {len(kpts1)})")

    except Exception as e:
        print(f"Error processing {main_img} vs {folder_img}: {str(e)}")
        continue

print("Processing completed!")
