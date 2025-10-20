import argparse
import os

import csv
import json
import torch

from ..LLaVA.llava.model.builder import load_pretrained_model

from .utils.conversation_utils import conv_templates
from .utils.image_utils import load_images
from .utils.llava_utils import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    disable_torch_init,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from .utils.utils import extract_json, set_seed
from .utils.video_utils import convert_video_to_grid


def eval_model(args):
    # preprocess: video to image grid
    image_grid_path = args.image_grid_path
    if image_grid_path == None:
        video_path = args.video_path
        image_grid_path = convert_video_to_grid(video_path)

    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    with open(args.read_prompt_file, "r") as json_data:
        prompts = json.load(json_data)

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    csv_path = os.path.join(output_path, f"{args.t2v_model}_consistent_attr_score.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as csvreader:
            reader = csv.reader(csvreader)
            lines = list(reader)  # Read all lines into a list
            line_count = len(lines)  # Count the number of lines
    else:
        line_count = 0

    with open(csv_path, "a", newline="") as csvfile:
        # Create a CSV writer
        csv_writer = csv.writer(csvfile)
        if line_count == 0:
            # Write the header row
            csv_writer.writerow(
                [
                    "name",
                    "prompt",
                    "seed0_answer1",
                    "seed0_answer2",
                    "seed0_answer3",
                    "seed0_score",
                    "seed1_answer1",
                    "seed1_answer2",
                    "seed1_answer3",
                    "seed1_score",
                    "seed2_answer1",
                    "seed2_answer2",
                    "seed2_answer3",
                    "seed2_score",
                    "seed_score",
                    "Score",
                ]
            )

    grid_images = [f for f in os.listdir(image_grid_path) if f[0].isdigit()]
    grid_images = sorted(grid_images)
    print(len(grid_images))

    evaluated = max(line_count - 1, 0)

    for i in range(evaluated, len(grid_images)):

        grid_image_name = grid_images[i]
        num = int(grid_image_name[0:4]) - 1

        phrases = prompts[num]["phrases"]
        this_prompt = prompts[num]["prompt"]
        phrase_1 = phrases.split(";")[0].strip()
        phrase_2 = phrases.split(";")[1].strip()

        image_files = [os.path.join(image_grid_path, grid_images[i])]
        images = load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(images, image_processor, model.config).to(
            model.device, dtype=torch.float16
        )

        Q1 = "The provided image arranges key frames from an AI generated video in a grid layout.  Describe the video, carefully examining objects rendering quality throughout the frames and their visual attributes."

        qs1 = Q1
        image_token_se = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )

        qs1 = DEFAULT_IMAGE_TOKEN + "\n" + qs1

        conv_mode = "chatml_direct"

        args.conv_mode = conv_mode

        conv_init = conv_templates[args.conv_mode].copy()
        conv_init.append_message(conv_init.roles[0], qs1)
        conv_init.append_message(conv_init.roles[1], None)
        prompt_init = conv_init.get_prompt()

        input_ids_init = (
            tokenizer_image_token(
                prompt_init, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        outputs_1 = []
        outputs_2 = []
        outputs_3 = []
        scores_tmp = []

        for iteration in range(3):
            set_seed(args.seed + iteration)

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs1)
            conv.append_message(conv.roles[1], None)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids_init,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,  # 1
                    max_new_tokens=args.max_new_tokens,  # 512
                    use_cache=True,
                )

            output_1 = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                0
            ].strip()
            outputs_1.append(output_1)
            conv.messages[-1][-1] = output_1

            Q2 = f"Please select one option from A to E for the question. \n \
Question: \n \
A: '{phrase_1}' is clearly portrayed in all the frames. \n \
B: '{phrase_1}' is present in some frames. \n \
C: '{phrase_1}' is not strictly portrayed (mix other feature). \n \
D: '{phrase_1}' is incorrectly portrayed (wrong feature). \n \
E: '{phrase_1}' is not present at all. \n \
Here is an example: the question evaluates the rendering of 'A green taxi'. If the green taxi is clearly absent in more than half frames, select B. If the taxi is both green and yellow, select C. If the taxi is not green, select D. If there is no taxi at all, select E.\n \
Select the most suitable option according to your previous description. \
Put the option in JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., C)."

            qs2 = Q2
            conv.append_message(conv.roles[0], qs2)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,  # 1
                    max_new_tokens=args.max_new_tokens,  # 512
                    use_cache=True,
                )

            output_2 = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                0
            ].strip()
            outputs_2.append(output_2)

            Q3 = f"Please select one option from A to E for the question. \n \
Question: \n \
A: '{phrase_2}' is clearly portrayed in all the frames. \n \
B: '{phrase_2}' is present in some frames. \n \
C: '{phrase_2}' is not strictly portrayed (mix other feature). \n \
D: '{phrase_2}' is incorrectly portrayed (wrong feature). \n \
E: '{phrase_2}' is not present at all. \n \
Here is an example: the question evaluates the rendering of 'A green taxi'. If the green taxi is clearly absent in more than half frames, select B. If the taxi is both green and yellow, select C. If the taxi is not green, select D. If there is no taxi at all, select E.\n \
Select the most suitable option according to your previous description. \
Put the option in JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., C)."

            qs3 = Q3
            conv.messages[-2][-1] = qs3
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,  # 1
                    max_new_tokens=args.max_new_tokens,  # 512
                    use_cache=True,
                )

            output_3 = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                0
            ].strip()
            outputs_3.append(output_3)

            try:
                json_obj_2 = extract_json(output_2)
                json_obj_3 = extract_json(output_3)
                option_value_2 = json_obj_2["adjust"]
                option_value_3 = json_obj_3["adjust"]
                option_value = f"{option_value_2}1,{option_value_3}2"
            except:
                option_value = "bad reply"

            if option_value in ["A1,A2", "A2,A1"]:
                score_tmp = 15
            elif option_value in ["A1,B2", "B1,A2", "A2,B1", "B2,A1"]:
                score_tmp = 14
            elif option_value in ["B1,B2", "B2,B1"]:
                score_tmp = 13
            elif option_value in ["A1,C2", "C1,A2", "A2,C1", "C2,A1"]:
                score_tmp = 12
            elif option_value in ["A1,D2", "D1,A2", "A2,D1", "D2,A1"]:
                score_tmp = 10
            elif option_value in ["A1,E2", "E1,A2", "A2,E1", "E2,A1"]:
                score_tmp = 8
            elif option_value in ["B1,C2", "C1,B2", "B2,C1", "C2,B1"]:
                score_tmp = 11
            elif option_value in ["B1,D2", "D1,B2", "B2,D1", "D2,B1"]:
                score_tmp = 9
            elif option_value in ["B1,E2", "E1,B2", "B2,E1", "E2,B1"]:
                score_tmp = 7
            elif option_value in ["C1,C2", "C2,C1"]:
                score_tmp = 6
            elif option_value in ["C1,D2", "D1,C2", "C2,D1", "D2,C1"]:
                score_tmp = 5
            elif option_value in ["C1,E2", "E1,C2", "C2,E1", "E2,C1"]:
                score_tmp = 3
            elif option_value in ["D1,D2", "D2,D1"]:
                score_tmp = 4
            elif option_value in ["D1,E2", "E1,D2", "D2,E1", "E2,D1"]:
                score_tmp = 2
            elif option_value in ["E1,E2", "E2,E1"]:
                score_tmp = 1
            else:
                score_tmp = "bad reply"
                print("reply wrong format")

            scores_tmp.append(score_tmp)

            print("score for", grid_images[i], score_tmp)

        int_flag = 0
        for score in scores_tmp:
            if not isinstance(score, int):
                int_flag = 1
        if int_flag == 0:
            score_avg = sum(scores_tmp) / len(scores_tmp)
        else:
            score_avg = "bad reply"

        csv_writer.writerow(
            [
                grid_image_name,
                this_prompt,
                outputs_1[0],
                outputs_2[0],
                outputs_3[0],
                scores_tmp[0],
                outputs_1[1],
                outputs_2[1],
                outputs_3[1],
                scores_tmp[1],
                outputs_1[2],
                outputs_2[2],
                outputs_3[2],
                scores_tmp[2],
                scores_tmp,
                score_avg,
            ]
        )
        csvfile.flush()

        return csv_path


def model_score(csv_path):
    with open(csv_path, "r") as file:
        reader = csv.reader(file)
        lines = list(reader)
        score = 0
        cnt = 0
        for line in lines[1:]:
            try:
                score_tmp = (float(line[-1]) - 1) / 14
                score += score_tmp
                cnt += 1
            except:
                continue

        score = score / cnt
        print(
            "number of images evaluated: ",
            cnt,
            " consistent attribute model score: ",
            score,
        )

    with open(csv_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["score: ", score])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.6-34b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--output-path",
        type=str,
        default="../csv_consistent_attr",
        help="path to store the video scores",
    )
    parser.add_argument(
        "--read-prompt-file",
        type=str,
        default="../meta_data/consistent_attribute_binding.json",
        help="path of txt file with input prompts and meta data",
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="path to videos",
    )
    parser.add_argument(
        "--t2v-model",
        type=str,
        required=True,
        help="model name",
    )

    parser.add_argument(
        "--image_grid_path",
        type=str,
        default=None,
        help="image grid path",
    )
    args = parser.parse_args()

    csv_path = eval_model(args)
    model_score(csv_path)
