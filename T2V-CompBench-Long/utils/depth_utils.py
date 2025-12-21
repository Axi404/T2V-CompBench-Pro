import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from DA.depth_anything.dpt import DepthAnything
from DA.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

import json
import cv2
from .video_utils import convert_video_to_frames


def run_depth(
    video_path,
    t2v_model,
    output_dir,
    meta_file,
    encoder="vitl",
    pred_only=True,
    grayscale=True,
    num_frames=16,
):

    margin_width = 50
    caption_height = 60

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    depth_anything = (
        DepthAnything.from_pretrained("LiheYoung/depth_anything_{}14".format(encoder))
        .to(DEVICE)
        .eval()
    )

    total_params = sum(param.numel() for param in depth_anything.parameters())
    print("Total parameters: {:.2f}M".format(total_params / 1e6))

    transform = Compose(
        [
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    with open(meta_file, "r") as json_data:
        prompts = json.load(json_data)

    frame_folder = convert_video_to_frames(video_path, num_frames=num_frames)
    print("frame_folder: ", frame_folder)

    output_dir = os.path.join(output_dir, t2v_model)
    videos = os.listdir(frame_folder)
    # Filter out non-numeric directories (e.g., 'frames')
    videos = [v for v in videos if v.isdigit()]
    videos.sort(key=lambda x: int(x))

    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(len(videos)), desc="Depth Estimation"):
        vid = videos[i]
        images = os.listdir(os.path.join(frame_folder, vid))
        images.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

        num = int(videos[i]) - 1

        prompt = prompts[num]["prompt"]
        spatial = prompts[num]["spatial"]  # A is on the left of B
        phrase_0 = prompts[num]["object_1"]
        phrase_1 = prompts[num]["object_2"]
        spatial = prompts[num]["spatial"]
        if spatial not in [
            "left",
            "right",
            "above",
            "on",
            "under",
            "below",
            "in front of",
            "behind",
        ]:
            print(spatial, "spatial not included!!!, index: ", vid)
            break

        if spatial in ["behind", "in front of"]:
            if pred_only:
                os.makedirs(os.path.join(output_dir, vid), exist_ok=True)

            # Preload all images and store metadata
            batch_size = 8  # Adjust based on GPU memory
            batch_images = []
            batch_meta = []  # (h, w, frame_name)

            for frame in images:
                filename = os.path.join(frame_folder, vid, frame)
                raw_image = cv2.imread(filename)
                image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
                h, w = image.shape[:2]
                image = transform({"image": image})["image"]
                batch_images.append(torch.from_numpy(image))
                batch_meta.append((h, w, frame))

            # Batch inference
            for batch_start in range(0, len(batch_images), batch_size):
                batch_end = min(batch_start + batch_size, len(batch_images))
                batch_tensors = torch.stack(batch_images[batch_start:batch_end]).to(
                    DEVICE
                )

                with torch.no_grad():
                    depths = depth_anything(batch_tensors)

                # Post-process each depth map in batch
                for idx in range(depths.shape[0]):
                    h, w, frame = batch_meta[batch_start + idx]
                    depth = depths[idx]
                    depth = F.interpolate(
                        depth[None, None], (h, w), mode="bilinear", align_corners=False
                    )[0, 0]
                    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                    depth = depth.cpu().numpy().astype(np.uint8)

                    if grayscale:
                        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                    else:
                        depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

                    if pred_only:
                        cv2.imwrite(os.path.join(output_dir, vid, frame), depth)
    return frame_folder
