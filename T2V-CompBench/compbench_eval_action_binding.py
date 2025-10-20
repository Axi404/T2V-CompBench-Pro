import argparse
import os

import torch
import csv
import json

from ..LLaVA.llava.model.builder import load_pretrained_model

from .utils.conversation_utils import conv_templates
from .utils.image_utils import load_images
from .utils.llava_utils import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    disable_torch_init,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from .utils.prompt_utils import (
    ACTION_BINDING_PROMPT_TEMPLATE_Q1 as Q1_template,
    ACTION_BINDING_PROMPT_TEMPLATE_Q2 as Q2_template,
    ACTION_BINDING_PROMPT_TEMPLATE_Q3_A as Q3_A_template,
    ACTION_BINDING_PROMPT_TEMPLATE_Q3_BC_OBJ1 as Q3_BC_obj1_template,
    ACTION_BINDING_PROMPT_TEMPLATE_Q3_BC_OBJ2 as Q3_BC_obj2_template,
)
from .utils.utils import (
    extract_json,
    initialize_csv,
    model_score,
    set_seed,
    write_to_csv,
)
from .utils.video_utils import convert_video_to_grid


def eval_model(args):
    # preprocess: video 2 grid
    image_grid_path = args.image_grid_path
    if image_grid_path == None:
        video_path = args.video_path
        image_grid_path = convert_video_to_grid(video_path)

    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    with open(args.read_prompt_file, "r") as json_data:
        prompts = json.load(json_data)

    csv_path, line_count = initialize_csv(
        args.output_path, args.t2v_model, "action_binding"
    )

    grid_images = [f for f in os.listdir(image_grid_path) if f[0].isdigit()]
    grid_images = sorted(grid_images)
    print("[INFO] number of images: ", len(grid_images))

    evaluated = max(line_count - 1, 0)

    for i in range(evaluated, len(grid_images)):
        # get image name
        grid_image_name = grid_images[i]
        num = int(grid_image_name[0:4]) - 1

        # prepare text inputs
        this_prompt = prompts[num]["prompt"]
        phrase_0 = prompts[num]["phrase_0"]  # get first obj and action in a list
        phrase_1 = prompts[num]["phrase_1"]  # get second obj and action in a list

        obj1 = phrase_0[0].split("?")[0]
        obj1_action = phrase_0[1].split("?")[0]
        obj2 = phrase_1[0].split("?")[0]
        obj2_action = phrase_1[1].split("?")[0]

        # prepare image inputs
        image_files = [os.path.join(image_grid_path, grid_images[i])]
        images = load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(images, image_processor, model.config).to(
            model.device, dtype=torch.float16
        )

        # prepare prompt templates
        Q1 = DEFAULT_IMAGE_TOKEN + "\n" + Q1_template
        Q2 = Q2_template.format(this_prompt=this_prompt, obj1=obj1, obj2=obj2)
        Q3_A = Q3_A_template.format(obj1_action=obj1_action, obj2_action=obj2_action)
        Q3_B = Q3_BC_obj1_template.format(obj1_action=obj1_action)
        Q3_C = Q3_BC_obj2_template.format(obj2_action=obj2_action)

        # prepare outputs volume
        outputs_1 = []
        outputs_2 = []
        outputs_3 = []
        scores_tmp = []

        for iteration in range(3):
            # set seed
            set_seed(args.seed + iteration)

            # conversation 1
            conv = conv_templates["chatml_direct"].copy()
            conv.append_message(conv.roles[0], Q1)
            conv.append_message(conv.roles[1], None)
            with torch.inference_mode():
                output_ids = model.generate(
                    tokenizer_image_token(
                        conv.get_prompt(),
                        tokenizer,
                        IMAGE_TOKEN_INDEX,
                        return_tensors="pt",
                    )
                    .unsqueeze(0)
                    .cuda(),
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

            # conversation 2
            conv.append_message(conv.roles[0], Q2)
            conv.append_message(conv.roles[1], None)
            with torch.inference_mode():
                output_ids = model.generate(
                    tokenizer_image_token(
                        conv.get_prompt(),
                        tokenizer,
                        IMAGE_TOKEN_INDEX,
                        return_tensors="pt",
                    )
                    .unsqueeze(0)
                    .cuda(),
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,  # 0.2
                    top_p=args.top_p,
                    num_beams=args.num_beams,  # 1
                    max_new_tokens=args.max_new_tokens,  # 512
                    use_cache=True,
                )
            output_2 = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                0
            ].strip()
            outputs_2.append(output_2)
            conv.messages[-1][-1] = output_2

            # parse model output
            json_obj_2 = extract_json(output_2)
            try:
                option_value_2 = json_obj_2["adjust"]
            except:
                option_value_2 = "bad reply 1"
            print("option_value_2 ", option_value_2)

            # whether to ask question 3
            if option_value_2 == "A":
                Q3 = Q3_A
                ask_Q3 = True
            elif option_value_2 == "B":
                Q3 = Q3_B
                ask_Q3 = True
            elif option_value_2 == "C":
                Q3 = Q3_C
                ask_Q3 = True
            elif option_value_2 == "D":
                score_tmp = 1
                ask_Q3 = False
            else:
                ask_Q3 = False
                score_tmp = "bad reply"
            if not ask_Q3:
                scores_tmp.append(score_tmp)
                outputs_3.append("")
                print("score for", grid_images[i], score_tmp)
                continue

            # conversation 3
            conv.append_message(conv.roles[0], Q3)
            conv.append_message(conv.roles[1], None)
            with torch.inference_mode():
                output_ids = model.generate(
                    tokenizer_image_token(
                        conv.get_prompt(),
                        tokenizer,
                        IMAGE_TOKEN_INDEX,
                        return_tensors="pt",
                    )
                    .unsqueeze(0)
                    .cuda(),
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,  # 0.2
                    top_p=args.top_p,
                    num_beams=args.num_beams,  # 1
                    max_new_tokens=args.max_new_tokens,  # 512
                    use_cache=True,
                )
            output_3 = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                0
            ].strip()
            outputs_3.append(output_3)

            # parse model output
            adjust_values = []
            for line in output_3.splitlines():
                if '"adjust":' in line:
                    # Extract the value after "adjust:"
                    value = line.split(":")[1].strip().strip('",')
                    adjust_values.append(value)
            option_value_3 = ",".join(adjust_values)
            print("option_value_3 ", option_value_3)

            # calculate score based on option_value_3
            if option_value_3 in ["A1,A2", "A2,A1"]:
                score_tmp = 10
            elif option_value_3 in ["A1,B2", "B1,A2", "A2,B1", "B2,A1"]:
                score_tmp = 9
            elif option_value_3 in ["A1,C2", "C1,A2", "A2,C1", "C2,A1"]:
                score_tmp = 8
            elif option_value_3 in ["B1,B2", "B2,B1"]:
                score_tmp = 7
            elif option_value_3 in ["B1,C2", "C1,B2", "B2,C1", "C2,B1"]:
                score_tmp = 6
            elif option_value_3 in ["C1,C2", "C2,C1"]:
                score_tmp = 5
            elif option_value_3 in ["A"]:
                score_tmp = 4
            elif option_value_3 in ["B"]:
                score_tmp = 3
            elif option_value_3 in ["C"]:
                score_tmp = 2
            else:
                score_tmp = "bad reply ?"
                print("reply wrong format")

            scores_tmp.append(score_tmp)
            print("score for", grid_images[i], score_tmp)

        # calculate average score
        has_bad_reply_flag = any(not isinstance(score, int) for score in scores_tmp)
        if has_bad_reply_flag:
            score_avg = "bad reply"
        else:
            score_avg = sum(scores_tmp) / len(scores_tmp)

        # write to csv
        write_to_csv(
            csv_path,
            benchmark_name="action_binding",
            grid_image_name=grid_image_name,
            this_prompt=this_prompt,
            outputs_1=outputs_1,
            outputs_2=outputs_2,
            outputs_3=outputs_3,
            scores_tmp=scores_tmp,
            score_avg=score_avg,
        )

    return csv_path


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
        default="../csv_action_binding",
        help="path to store the video scores",
    )
    parser.add_argument(
        "--read-prompt-file",
        type=str,
        default="../meta_data/action_binding.json",
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
