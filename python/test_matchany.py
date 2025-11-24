import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForKeypointMatching

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
image_dir = "/home/alex/Work/grad/bitirme/images"
output_dir = "/home/alex/Work/grad/bitirme/assets/matched_outputs/matchany/"
os.makedirs(output_dir, exist_ok=True)

# === Load processor and model ===
local_dir = "/home/alex/Work/grad/bitirme/matchany/matchanything_eloftr_local"  # contains config.json, model.safetensors, etc.
processor = AutoImageProcessor.from_pretrained(local_dir)
model = AutoModelForKeypointMatching.from_pretrained(local_dir).to(device)


# === Helper: Find matching pairs ===
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


# === Load and process ===
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
        image1 = Image.open(main_img_path).convert("RGB")
        image2 = Image.open(folder_img_path).convert("RGB")
    except Exception as e:
        print(f"Error loading images {main_img}, {folder_img}: {e}")
        continue

    # Run matching
    images = [image1, image2]
    inputs = processor(images, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process
    image_sizes = [[(img.height, img.width) for img in images]]
    matches = processor.post_process_keypoint_matching(
        outputs, image_sizes, threshold=0.2
    )

    # Extract keypoints and matches information
    if matches and len(matches) > 0:
        match_data = matches[0]
        keypoints0 = match_data.get("keypoints0", [])
        keypoints1 = match_data.get("keypoints1", [])
        matches0 = match_data.get("matches0", [])

        # Count valid matches (non-negative)
        num_matches = sum(1 for match in matches0 if match >= 0)
        num_keypoints0 = len(keypoints0)
        num_keypoints1 = len(keypoints1)
    else:
        num_matches = 0
        num_keypoints0 = 0
        num_keypoints1 = 0

    # Visualize
    vis_images = processor.visualize_keypoint_matching(images, matches)
    vis_image = vis_images[0]  # Single image

    # Create descriptive output filename
    main_base = os.path.splitext(main_img)[0]
    folder_img_name = os.path.splitext(os.path.basename(folder_img))[0]
    out_filename = f"{main_base}_vs_{folder_img_name}_matchany.png"
    save_path = os.path.join(output_dir, out_filename)

    # Save visualization
    vis_image.save(save_path)
    print(
        f"Saved: {save_path} (Keypoints: {num_keypoints0}:{num_keypoints1}, Matches: {num_matches})"
    )

print("Processing completed!")
