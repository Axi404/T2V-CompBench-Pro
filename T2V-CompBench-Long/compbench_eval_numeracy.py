import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from GSA.GroundingDINO.groundingdino.models import build_model
from GSA.GroundingDINO.groundingdino.util.slconfig import SLConfig
from GSA.GroundingDINO.groundingdino.util.utils import clean_state_dict

from utils.video_utils import convert_video_to_frames
from utils.relation_utils import filter_box
from utils.draw_utils import plot_boxes_to_image
from utils.image_utils import load_and_process_image
from utils.utils import initialize_csv, write_to_csv, combine_frame_numeracy
from utils.grounding_utils import get_grounding_output


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    print(load_res)
    _ = model.eval()
    return model


def process_single_image_with_single_object(
    model,
    image_path,
    objs,
    nums,
    box_threshold,
    text_threshold,
    iou_threshold,
    cpu_only=False,
):
    # load image
    image_pil, image = load_and_process_image(image_path)

    # run model
    image_with_box = image_pil
    all_box = []
    all_prob = []
    all_phrase = []
    for j in range(len(objs)):
        boxes_filt_0, pred_phrases_0, prob_0 = get_grounding_output(
            model,
            image,
            objs[j],
            box_threshold,
            text_threshold,
            # device="cpu" if cpu_only else "cuda",
        )
        size = image_pil.size
        all_box = all_box + [boxes_filt_0]
        all_prob = all_prob + prob_0
        all_phrase = all_phrase + pred_phrases_0

    all_box = torch.cat(all_box, dim=0)
    all_box, all_phrase, all_prob = filter_box(
        all_box, all_phrase, all_prob, iou_threshold
    )

    # plot boxes & probs on image
    pred_dict_0 = {
        "boxes": all_box,
        "size": [size[1], size[0]],  # H,W
        "labels": all_phrase,
    }
    image_with_box = plot_boxes_to_image(image_with_box, pred_dict_0, objs)[0]

    if len(objs) == 1:
        real_obj1_num = len(
            [
                phrase.split("(")[1]
                for phrase in all_phrase
                if phrase.split("(")[0] == objs[0]
            ]
        )
        real_obj_num = [real_obj1_num]
        score = 0
        if nums[0] == real_obj1_num:
            score += 1
    elif len(objs) > 1:
        real_obj1_num = len(
            [
                phrase.split("(")[1]
                for phrase in all_phrase
                if phrase.split("(")[0] == objs[0]
            ]
        )
        real_obj2_num = len(
            [
                phrase.split("(")[1]
                for phrase in all_phrase
                if phrase.split("(")[0] == objs[1]
            ]
        )
        real_obj_num = [real_obj1_num, real_obj2_num]
        score = 0
        if nums[0] == real_obj1_num:
            score += 0.5
        if nums[1] == real_obj2_num:
            score += 0.5
    else:
        raise ValueError(f"Invalid number of objects: {len(objs)}")
    return real_obj_num, score, image_with_box


def eval_model(args):
    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    output_dir = os.path.join(
        args.output_dir, args.t2v_model
    )  # change the path of the output image directory
    output_path = args.output_path  # change the path of the output score csv
    video_path = args.video_path  # change the path of the input video folder

    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    with open(args.read_prompt_file, "r") as json_data:
        prompts = json.load(json_data)

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    if args.frame_folder == None:
        frame_folder = convert_video_to_frames(video_path, num_frames=96)
    else:
        frame_folder = args.frame_folder

    videos = os.listdir(frame_folder)
    videos.sort()  # sort

    csv_path, line_count = initialize_csv(output_path, args.t2v_model, "numeracy_frame")
    evaluated = max(line_count - 1, 0)

    for i in tqdm(range(evaluated, len(videos)), desc="Numeracy Evaluation"):
        os.makedirs(os.path.join(output_dir, videos[i]), exist_ok=True)
        video_path = os.path.join(frame_folder, videos[i])
        images = os.listdir(video_path)
        images = [i for i in images if not os.path.isdir(os.path.join(video_path, i))]
        images.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))  # sort
        video_name = videos[i]
        num = int(video_name[0:4]) - 1

        prompt = prompts[num]["prompt"]
        objects = prompts[num]["objects"]
        numbers = prompts[num]["numbers"]
        objs = objects.split(",")
        nums = numbers.split(",")

        if len(objs) != len(nums):
            print("video ", i, " parse wrong, objects and quantities not match")
            break
        for j in range(len(objs)):
            objs[j] = objs[j].strip()
            try:
                nums[j] = nums[j].strip()
                nums[j] = int(nums[j])
            except:
                print("object number not int")
        print(objs, nums)

        for image_name in images:
            image_path = os.path.join(frame_folder, videos[i], image_name)
            real_obj_num, score, image_with_box = (
                process_single_image_with_single_object(
                    model,
                    image_path=image_path,
                    objs=objs,
                    nums=nums,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    iou_threshold=args.iou_threshold,
                    cpu_only=args.cpu_only,
                )
            )
            write_to_csv(
                csv_path,
                "numeracy_frame",
                video_name=videos[i],
                image_name=image_name,
                prompt=prompt,
                objects=objs,
                numbers=nums,
                actual_numbers=real_obj_num,
                score=score,
            )
            image_with_box.save(
                os.path.join(output_dir, videos[i], f"{image_name.split('.')[0]}.jpg")
            )
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        default="GSA/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="path to config file",
    )
    parser.add_argument(
        "--checkpoint_path",
        "-p",
        type=str,
        default="GSA/groundingdino_swint_ogc.pth",
        help="path to checkpoint file",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="playground/results/output_numeracy",
        help="directory to save the output images",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="playground/results/csv_numeracy",
        help="path to store the video scores",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.9,
        help="threshold to filter out the duplicated boxes",
    )
    parser.add_argument(
        "--box_threshold", type=float, default=0.4, help="box threshold"
    )
    parser.add_argument(
        "--text_threshold", type=float, default=0.25, help="text threshold"
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="running on cpu only!, default=False"
    )
    parser.add_argument(
        "--read-prompt-file",
        type=str,
        default="playground/meta_data/generative_numeracy.json",
        help="path to the meta data",
    )
    parser.add_argument(
        "--video-path", type=str, required=True, help="path to the input videos"
    )

    parser.add_argument(
        "--frame-folder", type=str, default=None, help="path to the frame folder"
    )
    parser.add_argument("--t2v-model", required=True, type=str)
    args = parser.parse_args()

    csv_path = eval_model(args)
    output_path = args.output_path
    combine_frame_numeracy(
        f"{output_path}/{args.t2v_model}_numeracy_frame.csv",
        f"{output_path}/{args.t2v_model}_numeracy_video.csv",
    )
