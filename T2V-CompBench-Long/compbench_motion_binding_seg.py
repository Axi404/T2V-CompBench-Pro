import argparse
import os
import sys

import cv2
import json
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import gc

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Grounding DINO
from GSA.GroundingDINO.groundingdino.models import build_model
from GSA.GroundingDINO.groundingdino.util.slconfig import SLConfig
from GSA.GroundingDINO.groundingdino.util.utils import clean_state_dict
from GSA.segment_anything.segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor,
)


from utils.draw_utils import show_mask, show_box
from utils.grounding_utils import get_grounding_output
from utils.image_utils import load_and_process_image
from utils.video_utils import convert_video_to_frames, convert_video_to_standard_video
from utils.utils import save_mask_foreground, save_mask_data


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    print(load_res)
    _ = model.eval()
    return model


def process_foreground_mask(
    obj_prompt: str,
    image_path: str,
    image_pil,
    image_loaded,
    model: torch.nn.Module,
    predictor: SamPredictor,
    box_threshold: float,
    text_threshold: float,
    output_dir: str,
    video_name: str,
    device: str,
) -> bool:
    """
    Process foreground mask for a single object.

    Args:
        obj_prompt: Object prompt for detection.
        image_path: Path to the image file.
        image_pil: PIL image object.
        image_loaded: Loaded and processed image tensor.
        model: Grounding model for object detection.
        predictor: SAM predictor for segmentation.
        box_threshold: Threshold for box detection.
        text_threshold: Threshold for text matching.
        output_dir: Directory to save output.
        video_name: Name of the video being processed.
        device: Device to run inference on.

    Returns:
        True if processing succeeded, False if no boxes detected.
    """
    boxes_filt, pred_phrases, probs = get_grounding_output(
        model,
        image_loaded,
        obj_prompt,
        box_threshold,
        text_threshold,
        device=device,
    )
    if boxes_filt.shape[0] == 0:
        return False

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes_filt, image.shape[:2]
    ).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=False)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)
    plt.axis("off")
    plt.savefig(
        os.path.join(output_dir, video_name, f"grounded_sam_output_{obj_prompt}.jpg"),
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.0,
    )
    plt.close()

    m = max(probs)
    ind = probs.index(m)
    mask_max_prob = masks[ind]
    save_mask_foreground(
        os.path.join(output_dir, video_name), mask_max_prob, obj_prompt
    )
    return True


def process_background_mask(
    background_prompt: str,
    image_path: str,
    image_pil,
    image_loaded,
    model: torch.nn.Module,
    predictor: SamPredictor,
    box_threshold: float,
    text_threshold: float,
    output_dir: str,
    video_name: str,
    device: str,
) -> bool:
    """
    Process background mask for the combined objects.

    Args:
        background_prompt: Combined prompt for background detection.
        image_path: Path to the image file.
        image_pil: PIL image object.
        image_loaded: Loaded and processed image tensor.
        model: Grounding model for object detection.
        predictor: SAM predictor for segmentation.
        box_threshold: Threshold for box detection.
        text_threshold: Threshold for text matching.
        output_dir: Directory to save output.
        video_name: Name of the video being processed.
        device: Device to run inference on.

    Returns:
        True if processing succeeded, False if no boxes detected.
    """
    boxes_filt, pred_phrases, probs = get_grounding_output(
        model,
        image_loaded,
        background_prompt,
        box_threshold,
        text_threshold,
        device=device,
    )
    if boxes_filt.shape[0] == 0:
        return False

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes_filt, image.shape[:2]
    ).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=False)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

    plt.axis("off")
    plt.savefig(
        os.path.join(output_dir, video_name, "grounded_sam_output_background.jpg"),
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.0,
    )
    plt.close()

    save_mask_data(
        os.path.join(output_dir, video_name), masks, boxes_filt, pred_phrases
    )  # save background
    return True


def process_single_video_segmentation(
    video_name: str,
    object_to_detect: list,
    background_prompt: str,
    frame_folder: str,
    output_dir: str,
    model: torch.nn.Module,
    predictor: SamPredictor,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> bool:
    """
    Process a single video for motion binding segmentation.

    Args:
        video_name: Name of the video being processed.
        object_to_detect: List of object prompts to detect.
        background_prompt: Combined prompt for background detection.
        frame_folder: Path to folder containing video frames.
        output_dir: Directory to save output.
        model: Grounding model for object detection.
        predictor: SAM predictor for segmentation.
        box_threshold: Threshold for box detection.
        text_threshold: Threshold for text matching.
        device: Device to run inference on.

    Returns:
        True if processing succeeded, False if background mask failed.
    """
    os.makedirs(os.path.join(output_dir, video_name), exist_ok=True)

    image_name = video_name + "_000000.png"
    image_path = os.path.join(frame_folder, video_name, image_name)

    # load image
    image_pil, image_loaded = load_and_process_image(image_path)

    # mask_foreground
    for obj_prompt in object_to_detect:
        process_foreground_mask(
            obj_prompt=obj_prompt,
            image_path=image_path,
            image_pil=image_pil,
            image_loaded=image_loaded,
            model=model,
            predictor=predictor,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            output_dir=output_dir,
            video_name=video_name,
            device=device,
        )

    # mask_background
    success = process_background_mask(
        background_prompt=background_prompt,
        image_path=image_path,
        image_pil=image_pil,
        image_loaded=image_loaded,
        model=model,
        predictor=predictor,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        output_dir=output_dir,
        video_name=video_name,
        device=device,
    )

    gc.collect()
    torch.cuda.empty_cache()
    return success


def foreground_background_mask(args):
    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    output_dir = os.path.join(args.output_dir, args.t2v_model)
    video_path = args.video_path
    total_frame = int(args.total_frame)
    fps = int(args.fps)

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(
            sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device)
        )
    else:
        predictor = SamPredictor(
            sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device)
        )

    frame_folder = convert_video_to_frames(video_path, num_frames=1)

    model_frames = (
        total_frame // fps * 8
    )  # number of frames to be extracted from the original video
    print(f"{args.t2v_model} extracted frames: {model_frames}")
    stardard_video_path = convert_video_to_standard_video(
        video_path, model_frames
    )  # take ~8 frames per second and recombine to a video of fps=8

    videos = os.listdir(frame_folder)
    videos.sort()

    with open(args.read_prompt_file, "r") as json_data:
        prompts = json.load(json_data)

    for k in tqdm(range(len(videos)), desc="Processing Motion Binding Segmentation"):

        video_name = videos[k]
        num = int(video_name[0:4]) - 1

        object_1 = prompts[num]["object_1"]
        object_2 = prompts[num]["object_2"]
        d_1 = prompts[num]["d_1"]
        d_2 = prompts[num]["d_2"]

        directions = ["left", "right", "up", "down", ""]
        if d_1 not in directions[:4] or d_2 not in directions:
            print(d_1, d_2, " direction not included!!!, index: ", k)
            break
        if object_2 != "":
            background_prompt = object_1 + " . " + object_2
            object_to_detect = [object_1, object_2]
        else:
            background_prompt = object_1
            object_to_detect = [object_1]

        process_single_video_segmentation(
            video_name=video_name,
            object_to_detect=object_to_detect,
            background_prompt=background_prompt,
            frame_folder=frame_folder,
            output_dir=output_dir,
            model=model,
            predictor=predictor,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )

    print("standard video path: ", stardard_video_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument(
        "--config",
        type=str,
        default="GSA/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="path to config file",
    )
    parser.add_argument(
        "--grounded_checkpoint",
        type=str,
        default="GSA/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        help="path to checkpoint file",
    )
    parser.add_argument(
        "--sam_version",
        type=str,
        default="vit_h",
        required=False,
        help="SAM ViT version: vit_b / vit_l / vit_h",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="GSA/sam_vit_h_4b8939.pth",
        help="path to sam checkpoint file",
    )
    parser.add_argument(
        "--sam_hq_checkpoint",
        type=str,
        default=None,
        help="path to sam-hq checkpoint file",
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )

    parser.add_argument(
        "--box_threshold", type=float, default=0.3, help="box threshold"
    )
    parser.add_argument(
        "--text_threshold", type=float, default=0.25, help="text threshold"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="running on cpu only!, default=False"
    )

    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--t2v-model", type=str, required=True)
    parser.add_argument("--total_frame", type=str, default=16, required=False)
    parser.add_argument("--fps", type=str, default=8, required=False)
    parser.add_argument(
        "--read-prompt-file",
        type=str,
        default="playground/meta_data/motion_binding.json",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="playground/results/output_motion_binding_seg",
        help="output directory of 1st frame segmentations",
    )
    args = parser.parse_args()

    foreground_background_mask(args)
