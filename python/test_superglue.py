# import os
# import re
# import torch
# import numpy as np
# from pathlib import Path
# from PIL import Image
# import matplotlib.cm as cm

# from models.matching import Matching
# from models.utils import make_matching_plot, read_image

# # === Config ===
# image_dir = '/home/alx/Documents/bitirme/scannet_sample_images'
# output_dir = os.path.join(os.getcwd(), 'assets', 'matched_outputs')
# os.makedirs(output_dir, exist_ok=True)

# resize = [640, 480]  # W x H
# resize_float = True
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # === Matcher config ===
# config = {
#     'superpoint': {
#         'nms_radius': 4,
#         'keypoint_threshold': 0.005,
#         'max_keypoints': 1024
#     },
#     'superglue': {
#         'weights': 'indoor',
#         'sinkhorn_iterations': 20,
#         'match_threshold': 0.2
#     }
# }
# matching = Matching(config).eval().to(device)

# # === Helper: group images by scene ID ===
# def group_images_by_scene(images):
#     pattern = re.compile(r'(scene\d{4}_\d{2})_frame-(\d+)\.jpg')
#     grouped = {}
#     for img in images:
#         match = pattern.match(img)
#         if match:
#             scene, frame = match.groups()
#             if scene not in grouped:
#                 grouped[scene] = []
#             grouped[scene].append((int(frame), img))
#     # Sort by frame
#     for scene in grouped:
#         grouped[scene] = sorted(grouped[scene])
#     return grouped

# # === Load image pairs ===
# image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
# grouped = group_images_by_scene(image_files)

# print(f"Found {len(grouped)} scenes.")

# for scene, frames in grouped.items():
#     if len(frames) < 2:
#         print(f"Skipping {scene}, only one image.")
#         continue

#     _, img1_name = frames[0]
#     _, img2_name = frames[1]

#     path1 = os.path.join(image_dir, img1_name)
#     path2 = os.path.join(image_dir, img2_name)

#     print(f"Processing: {img1_name} + {img2_name}")

#     image0, inp0, _ = read_image(path1, device, resize, 0, resize_float)
#     image1, inp1, _ = read_image(path2, device, resize, 0, resize_float)

#     if image0 is None or image1 is None:
#         print(f"Could not load images: {img1_name}, {img2_name}")
#         continue

#     with torch.no_grad():
#         pred = matching({'image0': inp0, 'image1': inp1})
#         pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

#     kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
#     matches = pred['matches0']
#     conf = pred['matching_scores0']

#     valid = matches > -1
#     mkpts0 = kpts0[valid]
#     mkpts1 = kpts1[matches[valid]]
#     mconf = conf[valid]
#     color = cm.jet(mconf)

#     # Save visualization
#     text = [
#         'SuperGlue',
#         f'Keypoints: {len(kpts0)}:{len(kpts1)}',
#         f'Matches: {len(mkpts0)}',
#     ]
#     small_text = [
#         f'Keypoint Threshold: {config["superpoint"]["keypoint_threshold"]}',
#         f'Match Threshold: {config["superglue"]["match_threshold"]}',
#         f'Pair: {Path(img1_name).stem}:{Path(img2_name).stem}',
#     ]
#     out_path = os.path.join(output_dir, f'{scene}_match_superglue.png')

#     make_matching_plot(
#         image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
#         text, out_path, show_keypoints=False,
#         fast_viz=False, opencv_display=False,
#         opencv_title='Matches', small_text=small_text
#     )

#     print(f"Saved: {out_path}")



import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.cm as cm

from models.matching import Matching
from models.utils import make_matching_plot, read_image

# === Config ===
image_dir = '/home/alex/Work/bitirme/images'
output_dir = os.path.join(os.getcwd(), 'assets', 'matched_outputs','superglue')
os.makedirs(output_dir, exist_ok=True)

resize = [640, 480]  # W x H
resize_float = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Matcher config ===
config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2
    }
}
matching = Matching(config).eval().to(device)

# === Helper: find matching pairs ===
def find_matching_pairs(image_dir):
    """Find all jpg files and their corresponding folders"""
    pairs = []
    
    # Get all jpg files in the directory
    jpg_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    
    for jpg_file in jpg_files:
        base_name = os.path.splitext(jpg_file)[0]  # Remove .jpg extension
        folder_path = os.path.join(image_dir, base_name)
        
        # Check if corresponding folder exists
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # Get all image files in the folder
            folder_images = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                folder_images.extend([f for f in os.listdir(folder_path) if f.lower().endswith(ext)])
            
            # Create pairs: (main_image, folder_image)
            for folder_img in folder_images:
                pairs.append((jpg_file, os.path.join(base_name, folder_img)))
    
    return pairs

# === Load image pairs ===
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

# === Process each pair ===
for main_img, folder_img in matching_pairs:
    main_img_path = os.path.join(image_dir, main_img)
    folder_img_path = os.path.join(image_dir, folder_img)
    
    print(f"Processing: {main_img} + {folder_img}")
    
    try:
        # Load images
        image0, inp0, _ = read_image(main_img_path, device, resize, 0, resize_float)
        image1, inp1, _ = read_image(folder_img_path, device, resize, 0, resize_float)

        if image0 is None or image1 is None:
            print(f"Could not load images: {main_img}, {folder_img}")
            continue

        # Perform matching
        with torch.no_grad():
            pred = matching({'image0': inp0, 'image1': inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches = pred['matches0']
        conf = pred['matching_scores0']

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        color = cm.jet(mconf)

        # Create descriptive output filename
        main_base = Path(main_img).stem
        folder_img_name = Path(folder_img).stem
        out_filename = f"{main_base}_vs_{folder_img_name}_superglue.png"
        out_path = os.path.join(output_dir, out_filename)

        # Prepare text for the visualization
        text = [
            'SuperGlue',
            f'Keypoints: {len(kpts0)}:{len(kpts1)}',
            f'Matches: {len(mkpts0)}',
        ]
        small_text = [
            f'Keypoint Threshold: {config["superpoint"]["keypoint_threshold"]}',
            f'Match Threshold: {config["superglue"]["match_threshold"]}',
            f'Pair: {main_base}:{folder_img_name}',
        ]

        # Save visualization
        make_matching_plot(
            image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
            text, out_path, show_keypoints=False,
            fast_viz=False, opencv_display=False,
            opencv_title='Matches', small_text=small_text
        )

        print(f"Saved: {out_path} (Matches: {len(mkpts0)})")

    except Exception as e:
        print(f"Error processing {main_img} vs {folder_img}: {str(e)}")
        continue

print("Processing completed!")