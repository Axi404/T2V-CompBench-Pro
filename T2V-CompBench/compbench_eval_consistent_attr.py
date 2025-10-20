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
from .utils.utils import (
    extract_json,
    initialize_csv,
    model_score,
    set_seed,
    write_to_csv,
)
from .utils.video_utils import convert_video_to_grid
from .utils.prompt_utils import (
    CONSISTENT_ATTR_PROMPT_TEMPLATE_Q1 as Q1_template,
    CONSISTENT_ATTR_PROMPT_TEMPLATE_Q2 as Q2_template,
    CONSISTENT_ATTR_PROMPT_TEMPLATE_Q3 as Q3_template,
)


def eval_model(args):
    # preprocess: video to image grid
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
        args.output_path, args.t2v_model, "consistent_attr"
    )

    grid_images = [f for f in os.listdir(image_grid_path) if f[0].isdigit()]
    grid_images = sorted(grid_images)
    print(len(grid_images))

    evaluated = max(line_count - 1, 0)

    for i in range(evaluated, len(grid_images)):
        # get image name
        grid_image_name = grid_images[i]
        num = int(grid_image_name[0:4]) - 1

        # prepare text inputs
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

        Q1 = DEFAULT_IMAGE_TOKEN + "\n" + Q1_template
        Q2 = Q2_template.format(phrase_1=phrase_1)
        Q3 = Q3_template.format(phrase_2=phrase_2)

        outputs_1 = []
        outputs_2 = []
        outputs_3 = []
        scores_tmp = []

        for iteration in range(3):
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

            # conversation 3
            conv.messages[-2][-1] = Q3
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
            output_3 = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                0
            ].strip()
            outputs_3.append(output_3)

            # parse model output
            try:
                json_obj_2 = extract_json(output_2)
                json_obj_3 = extract_json(output_3)
                option_value_2 = json_obj_2["adjust"]
                option_value_3 = json_obj_3["adjust"]
                option_value = f"{option_value_2}1,{option_value_3}2"
            except:
                option_value = "bad reply"

            # calculate score
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

        has_bad_reply_flag = any(not isinstance(score, int) for score in scores_tmp)
        if has_bad_reply_flag:
            score_avg = "bad reply"
        else:
            score_avg = sum(scores_tmp) / len(scores_tmp)

        write_to_csv(
            csv_path,
            benchmark_name="consistent_attr",
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
