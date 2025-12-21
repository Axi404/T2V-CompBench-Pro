import argparse
import os
from itertools import product
import json
import sys
import gc
from tqdm import tqdm

import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Grounding DINO
from DA.compbench_run_depth import run_depth
from GSA.GroundingDINO.groundingdino.models import build_model
from GSA.GroundingDINO.groundingdino.util.slconfig import SLConfig
from GSA.GroundingDINO.groundingdino.util.utils import clean_state_dict
from GSA.segment_anything.segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor,
)

from utils.relation_utils import (
    filter_box,
    spatial_judge,
    pick_max_2d,
    intersection_judge,
    pick_max_3d,
)
from utils.image_utils import load_and_process_image
from utils.draw_utils import plot_boxes_to_image, show_box, show_mask
from utils.utils import (
    combine_frame_spatial_relationships,
    combine_csv_and_cal_model_score,
    initialize_csv,
    write_to_csv,
)
from utils.grounding_utils import get_grounding_output

sys.path.append("./DA")


def load_model(model_config_path: str, model_checkpoint_path: str, device: str) -> torch.nn.Module:
    """
    Load and initialize the grounding model.
    
    Args:
        model_config_path: Path to model config file.
        model_checkpoint_path: Path to model checkpoint.
        device: Device to load model on ('cuda' or 'cpu').
    
    Returns:
        Loaded model on specified device.
    """
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    print(load_res)
    _ = model.eval()
    # Move model to device once during loading, avoid repeated .to(device) calls
    model = model.to(device)
    return model


def visualize_pred(
    image_pil,
    phrase_0,
    phrase_1,
    all_box,
    all_phrase,
    size,
    record_all_correct_spatial,
    selected_box_0,
    selected_box_1,
    selected_label,
    output_dir,
    video_name,
    image_name,
):
    pred_dict_0 = {
        "boxes": all_box,
        "size": [size[1], size[0]],  # H,W
        "labels": all_phrase,
    }
    color1 = (255, 200, 200)
    color2 = (150, 200, 255)
    image_with_box = plot_boxes_to_image(
        image_pil, pred_dict_0, [phrase_0, phrase_1], color1, color2
    )[0]

    if len(record_all_correct_spatial) != 0:
        # plot selected correct boxes
        pred_dict_1 = {
            "boxes": [selected_box_0, selected_box_1],
            "size": [size[1], size[0]],  # H,W
            "labels": selected_label,
        }
        color1 = (255, 0, 0)
        color2 = (0, 0, 255)
        image_with_box = plot_boxes_to_image(
            image_with_box,
            pred_dict_1,
            [phrase_0, phrase_1],
            color1,
            color2,
        )[0]

    os.makedirs(os.path.join(output_dir, video_name), exist_ok=True)
    image_with_box.save(os.path.join(output_dir, video_name, image_name))


def visualize_pred_3d(
    record_all_correct_spatial, mask0, mask1, output_dir, video_name, image_name
):
    if len(record_all_correct_spatial) != 0:
        color0 = np.array([150 / 255, 150 / 255, 255 / 255, 0.6])
        if mask0 is not None:
            show_mask(mask0.cpu().numpy(), plt.gca(), color0, random_color=False)
        color1 = np.array([255 / 255, 150 / 255, 150 / 255, 0.6])
        if mask1 is not None:
            show_mask(mask1.cpu().numpy(), plt.gca(), color1, random_color=False)

    plt.axis("off")
    os.makedirs(
        os.path.join(output_dir, video_name, image_name.split(".")[0]),
        exist_ok=True,
    )
    plt.savefig(
        os.path.join(
            output_dir,
            video_name,
            image_name.split(".")[0],
            f"grounded_sam_output.jpg",
        ),
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.0,
    )
    plt.close()


def get_cleaned_data(all_prob, all_phrase, all_box, phrase_0, phrase_1):
    clean_prob_0 = [
        all_prob[j]
        for j, phrase in enumerate(all_phrase)
        if phrase.split("(")[0] == phrase_0
    ]
    clean_prob_1 = [
        all_prob[j]
        for j, phrase in enumerate(all_phrase)
        if phrase.split("(")[0] == phrase_1
    ]
    clean_boxes_0 = [
        all_box[j]
        for j, phrase in enumerate(all_phrase)
        if phrase.split("(")[0] == phrase_0
    ]
    clean_boxes_1 = [
        all_box[j]
        for j, phrase in enumerate(all_phrase)
        if phrase.split("(")[0] == phrase_1
    ]
    clean_label_0 = [
        all_phrase[j]
        for j, phrase in enumerate(all_phrase)
        if phrase.split("(")[0] == phrase_0
    ]
    clean_label_1 = [
        all_phrase[j]
        for j, phrase in enumerate(all_phrase)
        if phrase.split("(")[0] == phrase_1
    ]
    return (
        clean_prob_0,
        clean_prob_1,
        clean_boxes_0,
        clean_boxes_1,
        clean_label_0,
        clean_label_1,
    )


def spatial_2d(args):
    torch.set_grad_enabled(False)
    config_file = args.config  # change the path of the model config file
    checkpoint_path = args.grounded_checkpoint  # change the path of the model
    output_dir = os.path.join(
        args.output_dir, args.t2v_model
    )  # change the path of the output image directory
    output_path = args.output_path  # change the path of the output score csv
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold_2d
    device = args.device

    # load model
    model = load_model(config_file, checkpoint_path, device=device)

    with open(args.read_prompt_file, "r") as json_data:
        prompts = json.load(json_data)

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    frame_folder = args.frame_folder
    videos = os.listdir(frame_folder)
    videos = [v for v in videos if not os.path.isdir(os.path.join(frame_folder, v))]
    videos.sort()  # sort

    csv_path, line_count = initialize_csv(args.output_path, args.t2v_model, "2dframe")
    for i in tqdm(range(len(videos)), desc="Evaluating Spatial 2D"):
        if i % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            plt.close("all")

        video_name = videos[i]
        num = int(video_name[0:4]) - 1

        prompt = prompts[num]["prompt"]
        spatial = prompts[num]["spatial"]  # A is on the left of B
        phrase_0 = prompts[num]["object_1"]
        phrase_1 = prompts[num]["object_2"]

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
            print(spatial, "spatial not included!!!, index: ", videos[i])
            continue

        if spatial in ["left", "right", "above", "on", "under", "below"]:
            os.makedirs(os.path.join(output_dir, videos[i]), exist_ok=True)
            video_path = os.path.join(frame_folder, videos[i])
            images = os.listdir(video_path)
            images.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))  # sort

            for image_name in images:
                image_path = os.path.join(frame_folder, videos[i], image_name)
                image_pil, image = load_and_process_image(image_path)

                # run model
                boxes_filt_0, pred_phrases_0, prob_0 = get_grounding_output(
                    model,
                    image,
                    phrase_0,
                    box_threshold,
                    text_threshold,
                    device=device,
                )
                boxes_filt_1, pred_phrases_1, prob_1 = get_grounding_output(
                    model,
                    image,
                    phrase_1,
                    box_threshold,
                    text_threshold,
                    device=device,
                )
                size = image_pil.size

                all_box = torch.cat((boxes_filt_0, boxes_filt_1), dim=0)
                all_prob = prob_0 + prob_1
                all_phrase = pred_phrases_0 + pred_phrases_1
                all_box, all_phrase, all_prob = filter_box(
                    all_box, all_phrase, all_prob, iou_threshold=iou_threshold
                )
                (
                    clean_prob_0,
                    clean_prob_1,
                    clean_boxes_0,
                    clean_boxes_1,
                    clean_label_0,
                    clean_label_1,
                ) = get_cleaned_data(all_prob, all_phrase, all_box, phrase_0, phrase_1)
                m0 = len(clean_prob_0)
                m1 = len(clean_prob_1)

                record_all_correct_spatial = []
                if m0 == 0 or m1 == 0:
                    if (m0 == 0 and m1 != 0) or (
                        m0 != 0 and m1 == 0
                    ):  # 1 object missing
                        score_1 = -1
                    elif m0 == 0 and m1 == 0:  # both objects missing
                        score_1 = -2
                    write_to_csv(
                        csv_path,
                        "2dframe",
                        video_name=videos[i],
                        image_name=image_name,
                        prompt=prompt,
                        m0=m0,
                        m1=m1,
                        score_1=score_1,
                    )
                    visualize_pred(
                        image_pil,
                        phrase_0,
                        phrase_1,
                        all_box,
                        all_phrase,
                        size,
                        record_all_correct_spatial,
                        None,
                        None,
                        None,
                        output_dir,
                        video_name,
                        image_name,
                    )
                    continue

                # if m0 != 0 and m1 != 0:
                for ii, jj in product(
                    range(len(clean_boxes_0)), range(len(clean_boxes_1))
                ):
                    _, correct_spatial, _, _, IoU, _ = spatial_judge(
                        clean_boxes_0[ii], clean_boxes_1[jj], spatial
                    )
                    if not correct_spatial:
                        continue

                    spatial_score_1 = 1 - IoU
                    prob_score_A = 0.5 * clean_prob_0[ii] + 0.5 * clean_prob_1[jj]
                    total_score_1 = 0.5 * spatial_score_1 + 0.5 * prob_score_A

                    info = {}
                    info["name"] = f"{ii}_{jj}"
                    info["box0"] = clean_boxes_0[ii]
                    info["box1"] = clean_boxes_1[jj]
                    info["total_score_1"] = total_score_1
                    info["spatial_score_1"] = spatial_score_1
                    info["label"] = [
                        clean_label_0[ii],
                        clean_label_1[jj],
                    ]
                    record_all_correct_spatial.append(info)

                if len(record_all_correct_spatial) != 0:
                    total_score_1_list = []
                    for candidate_box in record_all_correct_spatial:
                        total_score_1_list.append(candidate_box["total_score_1"])
                    score_1, selected_box_0, selected_box_1, selected_label = (
                        pick_max_2d(total_score_1_list, record_all_correct_spatial)
                    )
                else:
                    score_1 = 0  # wrong spatial relationship
                    selected_box_0 = None
                    selected_box_1 = None
                    selected_label = None

                write_to_csv(
                    csv_path,
                    "2dframe",
                    video_name=videos[i],
                    image_name=image_name,
                    prompt=prompt,
                    m0=m0,
                    m1=m1,
                    score_1=score_1,
                )

                # visualize pred
                visualize_pred(
                    image_pil,
                    phrase_0,
                    phrase_1,
                    all_box,
                    all_phrase,
                    size,
                    record_all_correct_spatial,
                    selected_box_0,
                    selected_box_1,
                    selected_label,
                    output_dir,
                    video_name,
                    image_name,
                )

    output_csv = combine_frame_spatial_relationships(
        f"{output_path}/{args.t2v_model}_2dframe.csv",
        f"{output_path}/{args.t2v_model}_2dvideo.csv",
    )
    return output_csv


def spatial_3d(args):
    torch.set_grad_enabled(False)
    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    depth_folder = os.path.join(args.depth_folder, args.t2v_model)

    output_dir = os.path.join(
        args.output_dir, args.t2v_model
    )  # change the path of the output image directory
    output_path = args.output_path  # change the path of the output score csv
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold_3d
    device = args.device

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

    with open(args.read_prompt_file, "r") as json_data:
        prompts = json.load(json_data)

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    frame_folder = args.frame_folder
    videos = os.listdir(frame_folder)
    videos = [v for v in videos if not os.path.isdir(os.path.join(frame_folder, v))]
    videos.sort()  # sort

    csv_path, line_count = initialize_csv(args.output_path, args.t2v_model, "3dframe")

    for i in tqdm(range(len(videos)), desc="Evaluating Spatial 3D"):
        video_name = videos[i]
        num = int(video_name[0:4]) - 1

        spatial = prompts[num]["spatial"]
        if spatial in ["in front of", "behind"]:
            prompt = prompts[num]["prompt"]
            phrase_0 = prompts[num]["object_1"]  # A is on the left of B
            phrase_1 = prompts[num]["object_2"]

            images = os.listdir(os.path.join(frame_folder, videos[i]))
            images.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

            for image_name in images:
                image_path = os.path.join(frame_folder, videos[i], image_name)

                # load image
                image_pil, image_loded = load_and_process_image(image_path)

                depth_path = os.path.join(depth_folder, videos[i], image_name)

                boxes_filt_0, pred_phrases_0, prob_0 = get_grounding_output(
                    model,
                    image_loded,
                    phrase_0,
                    box_threshold,
                    text_threshold,
                    device=device,
                )
                boxes_filt_1, pred_phrases_1, prob_1 = get_grounding_output(
                    model,
                    image_loded,
                    phrase_1,
                    box_threshold,
                    text_threshold,
                    device=device,
                )
                size = image_pil.size

                all_box = torch.cat((boxes_filt_0, boxes_filt_1), dim=0)
                all_prob = prob_0 + prob_1
                all_phrase = pred_phrases_0 + pred_phrases_1
                all_box, all_phrase, all_prob = filter_box(
                    all_box, all_phrase, all_prob, iou_threshold=iou_threshold
                )

                (
                    clean_prob_0,
                    clean_prob_1,
                    clean_boxes_0,
                    clean_boxes_1,
                    clean_label_0,
                    clean_label_1,
                ) = get_cleaned_data(all_prob, all_phrase, all_box, phrase_0, phrase_1)
                if len(clean_boxes_0) > 0:
                    boxes_filt_0 = torch.stack(clean_boxes_0, dim=0)
                else:
                    boxes_filt_0 = torch.tensor([])
                if len(clean_boxes_1) > 0:
                    boxes_filt_1 = torch.stack(clean_boxes_1, dim=0)
                else:
                    boxes_filt_1 = torch.tensor([])
                m0 = len(clean_prob_0)
                m1 = len(clean_prob_1)

                # sam
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                predictor.set_image(image)

                H, W = size[1], size[0]
                for k in range(boxes_filt_0.size(0)):
                    boxes_filt_0[k] = boxes_filt_0[k] * torch.Tensor([W, H, W, H])
                    boxes_filt_0[k][:2] -= boxes_filt_0[k][2:] / 2
                    boxes_filt_0[k][2:] += boxes_filt_0[k][:2]
                boxes_filt_0 = boxes_filt_0.cpu()

                for k in range(boxes_filt_1.size(0)):
                    boxes_filt_1[k] = boxes_filt_1[k] * torch.Tensor([W, H, W, H])
                    boxes_filt_1[k][:2] -= boxes_filt_1[k][2:] / 2
                    boxes_filt_1[k][2:] += boxes_filt_1[k][:2]
                boxes_filt_1 = boxes_filt_1.cpu()

                transformed_boxes_0 = predictor.transform.apply_boxes_torch(
                    boxes_filt_0, image.shape[:2]
                ).to(device)
                transformed_boxes_1 = predictor.transform.apply_boxes_torch(
                    boxes_filt_1, image.shape[:2]
                ).to(device)

                if m0 != 0:
                    masks_0, _, _ = predictor.predict_torch(  # masks_0[0]:[1,320,576]
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes_0.to(device),
                        multimask_output=False,
                    )

                if m1 != 0:
                    masks_1, _, _ = predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes_1.to(device),
                        multimask_output=False,
                    )

                record_all_correct_spatial = []

                if m0 == 0 or m1 == 0:
                    if (m0 == 0 and m1 != 0) or (m0 != 0 and m1 == 0):
                        score_1 = -1
                    elif m0 == 0 and m1 == 0:
                        score_1 = -2
                    write_to_csv(
                        csv_path,
                        "3dframe",
                        video_name=videos[i],
                        image_name=image_name,
                        prompt=prompt,
                        m0=m0,
                        m1=m1,
                        score_1=score_1,
                    )

                    # visualize pred
                    visualize_pred_3d(
                        record_all_correct_spatial,
                        None,
                        None,
                        output_dir,
                        videos[i],
                        image_name,
                    )
                    continue

                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                for box, label in zip(boxes_filt_0, clean_label_0):
                    show_box(box.numpy(), plt.gca(), label)
                for box, label in zip(boxes_filt_1, clean_label_1):
                    show_box(box.numpy(), plt.gca(), label)

                for ii, jj in product(
                    range(len(clean_boxes_0)), range(len(clean_boxes_1))
                ):
                    IoU, _ = intersection_judge(clean_boxes_0[ii], clean_boxes_1[jj])
                    if IoU != 0:
                        depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

                        mask_image_0 = (
                            masks_0[ii].cpu().numpy().squeeze() * 255
                        ).astype(np.uint8)
                        obj1_seg = cv2.bitwise_and(
                            depth_map, depth_map, mask=mask_image_0
                        )
                        non_zero_0 = cv2.countNonZero(mask_image_0)
                        if non_zero_0 == 0:
                            d1 = 0
                        else:
                            d1 = np.sum(obj1_seg) / non_zero_0

                        mask_image_1 = (
                            masks_1[jj].cpu().numpy().squeeze() * 255
                        ).astype(np.uint8)
                        obj2_seg = cv2.bitwise_and(
                            depth_map, depth_map, mask=mask_image_1
                        )
                        non_zero_1 = cv2.countNonZero(mask_image_1)
                        if non_zero_1 == 0:
                            d2 = 0
                        else:
                            d2 = np.sum(obj2_seg) / non_zero_1

                        if (not 0 <= d1 <= 255) or (not 0 <= d2 <= 255):
                            print("d1 wrong value")
                        if spatial == "in front of":
                            if d1 > d2:
                                prob_score = 0.5 * prob_0[ii] + 0.5 * prob_1[jj]
                                spatial_score_1 = IoU
                                total_score_1 = 0.5 * prob_score + 0.5 * spatial_score_1

                                seg_save_path = os.path.join(
                                    output_dir,
                                    videos[i],
                                    image_name.split(".")[0],
                                    f"obj1_seg_{ii}.png",
                                )
                                cv2.imwrite(seg_save_path, obj1_seg)
                                seg_save_path = os.path.join(
                                    output_dir,
                                    videos[i],
                                    image_name.split(".")[0],
                                    f"obj2_seg_{jj}.png",
                                )
                                cv2.imwrite(seg_save_path, obj2_seg)

                                info = {}
                                info["name"] = f"{ii}_{jj}"
                                info["box0"] = clean_boxes_0[ii]
                                info["box1"] = clean_boxes_1[jj]
                                info["total_score_1"] = total_score_1
                                info["spatial_score_1"] = spatial_score_1
                                info["mask0"] = masks_0[ii]
                                info["mask1"] = masks_1[jj]

                                record_all_correct_spatial.append(info)

                        elif spatial == "behind":
                            if d1 < d2:
                                prob_score = 0.5 * prob_0[ii] + 0.5 * prob_1[jj]
                                spatial_score_1 = IoU
                                total_score_1 = 0.5 * prob_score + 0.5 * spatial_score_1

                                seg_save_path = os.path.join(
                                    output_dir,
                                    videos[i],
                                    image_name.split(".")[0],
                                    f"obj1_seg_{ii}.png",
                                )
                                cv2.imwrite(seg_save_path, obj1_seg)
                                seg_save_path = os.path.join(
                                    output_dir,
                                    videos[i],
                                    image_name.split(".")[0],
                                    f"obj2_seg_{jj}.png",
                                )
                                cv2.imwrite(seg_save_path, obj2_seg)

                                info = {}
                                info["name"] = f"{ii}_{jj}"
                                info["box0"] = clean_boxes_0[ii]
                                info["box1"] = clean_boxes_1[jj]
                                info["total_score_1"] = total_score_1
                                info["spatial_score_1"] = spatial_score_1
                                info["mask0"] = masks_0[ii]
                                info["mask1"] = masks_1[jj]

                                record_all_correct_spatial.append(info)

                if len(record_all_correct_spatial) != 0:
                    total_score_1_list = []
                    for candidate_box in record_all_correct_spatial:
                        total_score_1_list.append(candidate_box["total_score_1"])
                    score_1, mask0, mask1 = pick_max_3d(
                        total_score_1_list, record_all_correct_spatial
                    )

                else:
                    score_1 = 0
                    mask0 = None
                    mask1 = None

                write_to_csv(
                    csv_path,
                    "3dframe",
                    video_name=videos[i],
                    image_name=image_name,
                    prompt=prompt,
                    m0=m0,
                    m1=m1,
                    score_1=score_1,
                )

                # visualize pred
                visualize_pred_3d(
                    record_all_correct_spatial,
                    mask0,
                    mask1,
                    output_dir,
                    videos[i],
                    image_name,
                )
    output_csv = combine_frame_spatial_relationships(
        f"{output_path}/{args.t2v_model}_3dframe.csv",
        f"{output_path}/{args.t2v_model}_3dvideo.csv",
    )
    return output_csv


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    # 2d args
    parser.add_argument(
        "--config",
        type=str,
        default="GSA/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="path to config file",
    )
    parser.add_argument(
        "--grounded_checkpoint",
        type=str,
        default="GSA/groundingdino_swint_ogc.pth",
        help="path to checkpoint file",
    )
    parser.add_argument(
        "--box_threshold", type=float, default=0.35, help="box threshold"
    )
    parser.add_argument(
        "--text_threshold", type=float, default=0.25, help="text threshold"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="running on cpu only!, default=False"
    )

    parser.add_argument(
        "--iou_threshold_2d",
        type=float,
        default=0.9,
        help="threshold to filter out the duplicated boxes",
    )

    # 3d args
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

    parser.add_argument("--depth_folder", type=str, required=False, default="playground/results/output_spatial_depth")
    parser.add_argument(
        "--iou_threshold_3d",
        type=float,
        default=0.95,
        help="threshold to filter out the duplicated boxes",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="playground/results/csv_spatial",
        help="path to store the video scores",
    )
    parser.add_argument(
        "--read-prompt-file", type=str, default="playground/meta_data/spatial_relationships.json"
    )
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--frame_folder", type=str)
    parser.add_argument("--t2v-model", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="playground/results/output_spatial/",
        help="directory to save the output images",
    )

    args = parser.parse_args()

    frame_folder = run_depth(
        args.video_path, args.t2v_model, args.depth_folder, args.read_prompt_file, num_frames=96
    )

    args.frame_folder = frame_folder
    csv_2d = spatial_2d(args)
    csv_3d = spatial_3d(args)
    output_csv = f"{args.output_path}/{args.t2v_model}_spatial_score.csv"
    combine_csv_and_cal_model_score(csv_2d, csv_3d, output_csv)
