# # # from transformers import AutoImageProcessor, AutoModelForKeypointMatching
# # # from transformers.image_utils import load_image
# # # import torch

# # # # Load a pair of images
# # # image1 = load_image("https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg")
# # # image2 = load_image("https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg")

# # # images = [image1, image2]

# # # # Load the processor and model from the Hugging Face Hub
# # # processor = AutoImageProcessor.from_pretrained("zju-community/matchanything_eloftr")
# # # model = AutoModelForKeypointMatching.from_pretrained("zju-community/matchanything_eloftr")

# # # # Process images and get model outputs
# # # inputs = processor(images, return_tensors="pt")
# # # with torch.no_grad():
# # #     outputs = model(**inputs)


# # # image_sizes = [[(image.height, image.width) for image in images]]
# # # outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
# # # for i, output in enumerate(outputs):
# # #     print("For the image pair", i)
# # #     for keypoint0, keypoint1, matching_score in zip(
# # #             output["keypoints0"], output["keypoints1"], output["matching_scores"]
# # #     ):
# # #         print(
# # #             f"Keypoint at coordinate {keypoint0.numpy()} in the first image matches with keypoint at coordinate {keypoint1.numpy()} in the second image with a score of {matching_score}."
# # #         )


# # # # plot_images = processor.visualize_keypoint_matching(images, outputs)

# # # import matplotlib.pyplot as plt
# # # vis_images = processor.visualize_keypoint_matching(images, outputs)
# # # single_image = vis_images[0]

# # # plt.figure(figsize=(15, 10))
# # # plt.imshow(single_image)
# # # plt.axis('off')
# # # plt.show()



# # # from transformers import AutoImageProcessor, AutoModelForKeypointMatching
# # # from transformers.image_utils import load_image
# # # import torch
# # # import matplotlib.pyplot as plt

# # # # Device setup: use GPU if available
# # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # print(f"Using device: {device}")

# # # # Load a pair of images
# # # image1 = load_image(
# # #     "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
# # # )
# # # image2 = load_image(
# # #     "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
# # # )
# # # images = [image1, image2]

# # # # Load processor and model from local directory
# # # local_dir = "./matchanything_eloftr_local"  # your folder with model.safetensors
# # # processor = AutoImageProcessor.from_pretrained(local_dir)
# # # model = AutoModelForKeypointMatching.from_pretrained(local_dir).to(device)

# # # # Process images
# # # inputs = processor(images, return_tensors="pt").to(device)
# # # with torch.no_grad():
# # #     outputs = model(**inputs)

# # # # Post-process keypoints
# # # image_sizes = [[(image.height, image.width) for image in images]]
# # # matches = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)

# # # # Print match info
# # # # for i, output in enumerate(matches):
# # # #     print(f"\nMatches for image pair {i}:")
# # # #     for k0, k1, score in zip(
# # # #         output["keypoints0"], output["keypoints1"], output["matching_scores"]
# # # #     ):
# # # #         print(f"{k0.numpy()} → {k1.numpy()} (score: {score:.2f})")

# # # # Visualize matches
# # # vis_images = processor.visualize_keypoint_matching(images, matches)
# # # single_image = vis_images[0]  # extract the first image from batch

# # # plt.figure(figsize=(15, 10))
# # # plt.imshow(single_image)
# # # plt.axis('off')
# # # plt.show()



# # import os
# # import re
# # import torch
# # import matplotlib.pyplot as plt
# # from PIL import Image
# # from transformers import AutoImageProcessor, AutoModelForKeypointMatching

# # # === Setup ===
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print(f"Using device: {device}")

# # # Paths
# # image_dir = "/home/alx/Documents/bitirme/scannet_sample_images"  # where your image pairs are
# # output_dir = os.path.join(os.getcwd(), "assets", "matched_outputs")
# # os.makedirs(output_dir, exist_ok=True)

# # # === Load processor and model ===
# # local_dir = "./matchanything_eloftr_local"  # contains config.json, model.safetensors, etc.
# # processor = AutoImageProcessor.from_pretrained(local_dir)
# # model = AutoModelForKeypointMatching.from_pretrained(local_dir).to(device)

# # # === Helper: Group images by scene prefix ===
# # def group_images_by_scene(images):
# #     pattern = re.compile(r'(scene\d{4}_\d{2})_frame-(\d+)\.jpg')
# #     grouped = {}
# #     for img in images:
# #         match = pattern.match(img)
# #         if match:
# #             scene, frame = match.groups()
# #             if scene not in grouped:
# #                 grouped[scene] = []
# #             grouped[scene].append((int(frame), img))
# #     # Sort by frame index
# #     for scene in grouped:
# #         grouped[scene] = sorted(grouped[scene])
# #     return grouped

# # # === Load and process ===
# # image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
# # grouped = group_images_by_scene(image_files)

# # print(f"Found {len(grouped)} scenes.")

# # for scene, frames in grouped.items():
# #     if len(frames) < 2:
# #         print(f"Skipping {scene}, only one image.")
# #         continue

# #     _, img1_name = frames[0]
# #     _, img2_name = frames[1]

# #     path1 = os.path.join(image_dir, img1_name)
# #     path2 = os.path.join(image_dir, img2_name)

# #     try:
# #         image1 = Image.open(path1).convert("RGB")
# #         image2 = Image.open(path2).convert("RGB")
# #     except Exception as e:
# #         print(f"Error loading images {img1_name}, {img2_name}: {e}")
# #         continue

# #     print(f"Processing {scene}: {img1_name} + {img2_name}")

# #     # Run matching
# #     images = [image1, image2]
# #     inputs = processor(images, return_tensors="pt").to(device)
# #     with torch.no_grad():
# #         outputs = model(**inputs)

# #     # Post-process
# #     image_sizes = [[(img.height, img.width) for img in images]]
# #     matches = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)

# #     # Visualize
# #     vis_images = processor.visualize_keypoint_matching(images, matches)
# #     vis_image = vis_images[0]  # Single image

# #     # Save visualization
# #     save_path = os.path.join(output_dir, f"{scene}_match_transformers.png")
# #     vis_image.save(save_path)
# #     print(f"Saved: {save_path}")


# import os
# import torch
# import matplotlib.pyplot as plt
# from PIL import Image
# from transformers import AutoImageProcessor, AutoModelForKeypointMatching

# # === Setup ===
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Paths
# image_dir = "/home/alx/Documents/bitirme/images"
# output_dir = "/home/alx/Documents/bitirme/assets/matched_outputs"
# os.makedirs(output_dir, exist_ok=True)

# # === Load processor and model ===
# local_dir = "./matchanything_eloftr_local"  # contains config.json, model.safetensors, etc.
# processor = AutoImageProcessor.from_pretrained(local_dir)
# model = AutoModelForKeypointMatching.from_pretrained(local_dir).to(device)

# # === Helper: Find matching pairs ===
# def find_matching_pairs(image_dir):
#     """Find all jpg files and their corresponding folders"""
#     pairs = []
    
#     # Get all jpg files in the directory
#     jpg_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    
#     for jpg_file in jpg_files:
#         base_name = os.path.splitext(jpg_file)[0]  # Remove .jpg extension
#         folder_path = os.path.join(image_dir, base_name)
        
#         # Check if corresponding folder exists
#         if os.path.exists(folder_path) and os.path.isdir(folder_path):
#             # Get all image files in the folder
#             folder_images = []
#             for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
#                 folder_images.extend([f for f in os.listdir(folder_path) if f.lower().endswith(ext)])
            
#             # Create pairs: (main_image, folder_image)
#             for folder_img in folder_images:
#                 pairs.append((jpg_file, os.path.join(base_name, folder_img)))
    
#     return pairs

# # === Load and process ===
# matching_pairs = find_matching_pairs(image_dir)
# print(f"Found {len(matching_pairs)} image pairs to process.")

# if not matching_pairs:
#     print("No matching pairs found. Please check your directory structure.")
#     print("Expected structure:")
#     print("image_dir/")
#     print("├── x.jpg")
#     print("├── x/")
#     print("│   ├── image1.jpg")
#     print("│   ├── image2.jpg")
#     print("│   └── ...")
#     print("├── y.jpg")
#     print("├── y/")
#     print("│   ├── image1.jpg")
#     print("│   └── ...")

# # Process each pair
# for main_img, folder_img in matching_pairs:
#     main_img_path = os.path.join(image_dir, main_img)
#     folder_img_path = os.path.join(image_dir, folder_img)
    
#     print(f"Processing: {main_img} + {folder_img}")
    
#     try:
#         # Load images
#         image1 = Image.open(main_img_path).convert("RGB")
#         image2 = Image.open(folder_img_path).convert("RGB")
#     except Exception as e:
#         print(f"Error loading images {main_img}, {folder_img}: {e}")
#         continue

#     # Run matching
#     images = [image1, image2]
#     inputs = processor(images, return_tensors="pt").to(device)
    
#     with torch.no_grad():
#         outputs = model(**inputs)

#     # Post-process
#     image_sizes = [[(img.height, img.width) for img in images]]
#     matches = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)

#     # Visualize
#     vis_images = processor.visualize_keypoint_matching(images, matches)
#     vis_image = vis_images[0]  # Single image

#     # Create descriptive output filename
#     main_base = os.path.splitext(main_img)[0]
#     folder_img_name = os.path.splitext(os.path.basename(folder_img))[0]
#     out_filename = f"{main_base}_vs_{folder_img_name}_transformers.png"
#     save_path = os.path.join(output_dir, out_filename)

#     # Save visualization
#     vis_image.save(save_path)
#     print(f"Saved: {save_path}")

# print("Processing completed!")



import os
import torch
import matplotlib.pyplot as plt
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
    matches = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)

    # Extract keypoints and matches information
    if matches and len(matches) > 0:
        match_data = matches[0]
        keypoints0 = match_data.get('keypoints0', [])
        keypoints1 = match_data.get('keypoints1', [])
        matches0 = match_data.get('matches0', [])
        
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
    print(f"Saved: {save_path} (Keypoints: {num_keypoints0}:{num_keypoints1}, Matches: {num_matches})")

print("Processing completed!")
