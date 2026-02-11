import argparse
import json
import os
import sys
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.qwen3_utils import (
    assistant_message,
    generate_with_messages,
    image_message,
    load_qwen3_model,
    text_message,
)
from utils.prompt_utils import (
    INTERACTION_PROMPT_TEMPLATE_Q1 as Q1_template,
    INTERACTION_PROMPT_TEMPLATE_Q2 as Q2_template,
    INTERACTION_PROMPT_TEMPLATE_Q3_A as Q3_A_template,
    INTERACTION_PROMPT_TEMPLATE_Q3_B as Q3_B_template,
    INTERACTION_PROMPT_TEMPLATE_Q3_C as Q3_C_template,
)
from utils.utils import (
    extract_json,
    initialize_csv,
    set_seed,
    write_to_csv,
)
from utils.video_utils import convert_video_to_grid


def run_interaction_conversation(
    model,
    processor,
    image_path: str,
    Q1: str,
    Q2: str,
    Q3_A: str,
    Q3_B: str,
    Q3_C: str,
    args,
    image_name: str = "",
) -> tuple[str, str, str, int | str]:
    """
    Run a single iteration of the object interaction multi-turn conversation.

    Args:
        model: The Qwen3-VL model
        processor: The processor
        image_path: Input image path
        Q1: First question prompt
        Q2: Second question prompt
        Q3_A: Third question prompt (option A)
        Q3_B: Third question prompt (option B)
        Q3_C: Third question prompt (option C)
        args: Arguments containing temperature, top_p, num_beams, max_new_tokens
        image_name: Image name for logging

    Returns:
        tuple: (output_1, output_2, output_3, score_tmp)
    """
    kw = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
    )

    messages = [image_message(image_path, Q1)]
    output_1 = generate_with_messages(model, processor, messages, **kw)

    messages += [assistant_message(output_1), text_message(Q2)]
    output_2 = generate_with_messages(model, processor, messages, **kw)

    # parse model output
    json_obj_2 = extract_json(output_2)
    try:
        option_value_2 = json_obj_2["adjust"]
    except:
        option_value_2 = "bad reply 1"
    print("option_value_2 ", option_value_2)

    # set conversation 3
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
        print("score for", image_name, score_tmp)
        return output_1, output_2, "", score_tmp

    messages += [assistant_message(output_2), text_message(Q3)]
    output_3 = generate_with_messages(model, processor, messages, **kw)

    # parse model output
    json_obj_3 = extract_json(output_3)
    try:
        option_value_3 = json_obj_3["adjust"]
    except:
        option_value_3 = "bad reply 2"
    print("option_value_3 ", option_value_3)

    # calculate score
    if option_value_2 == "A":
        if option_value_3 == "A":
            score_tmp = 10
        elif option_value_3 == "B":
            score_tmp = 9
        elif option_value_3 == "C":
            score_tmp = 8
        elif option_value_3 == "D":
            score_tmp = 7
        elif option_value_3 == "E":
            score_tmp = 6
        else:
            score_tmp = "bad reply ?"
            print("reply wrong format")
    elif option_value_2 == "B":
        if option_value_3 == "A":
            score_tmp = 5
        elif option_value_3 == "B":
            score_tmp = 3
        else:
            score_tmp = "bad reply ?"
            print("reply wrong format")
    elif option_value_2 == "C":
        if option_value_3 == "A":
            score_tmp = 4
        elif option_value_3 == "B":
            score_tmp = 2
        else:
            score_tmp = "bad reply ?"
            print("reply wrong format")

    print("score for", image_name, score_tmp)
    return output_1, output_2, output_3, score_tmp


def eval_model(args):
    # preprocess: video 2 grid
    image_grid_path = args.image_grid_path
    if image_grid_path == None:
        video_path = args.video_path
        image_grid_path = convert_video_to_grid(video_path)

    # Model
    if args.model_base is not None or args.conv_mode is not None or args.sep != ",":
        print("[WARN] --model-base/--conv-mode/--sep are deprecated and ignored.")
    model, processor = load_qwen3_model(args.model_path)
    with open(args.read_prompt_file, "r") as json_data:
        prompts = json.load(json_data)

    csv_path, line_count = initialize_csv(
        args.output_path, args.t2v_model, "object_interaction_score"
    )
    evaluated = max(line_count - 1, 0)

    grid_images = [
        f
        for f in os.listdir(image_grid_path)
        if not os.path.isdir(os.path.join(image_grid_path, f))
    ]
    grid_images = sorted(grid_images)
    print(len(grid_images))

    for i in range(evaluated, len(grid_images)):
        # get image name
        grid_image_name = grid_images[i]

        num = int(grid_image_name[0:4]) - 1

        this_prompt = prompts[num]["prompt"]

        image_path = os.path.join(image_grid_path, grid_images[i])

        Q1 = Q1_template
        Q2 = Q2_template.format(this_prompt=this_prompt)
        Q3_A = Q3_A_template.format(this_prompt=this_prompt)
        Q3_B = Q3_B_template.format(this_prompt=this_prompt)
        Q3_C = Q3_C_template.format(this_prompt=this_prompt)

        outputs_1 = []
        outputs_2 = []
        outputs_3 = []
        scores_tmp = []

        for iteration in range(3):
            set_seed(args.seed + iteration)

            # run multi-turn conversation
            output_1, output_2, output_3, score_tmp = run_interaction_conversation(
                model=model,
                processor=processor,
                image_path=image_path,
                Q1=Q1,
                Q2=Q2,
                Q3_A=Q3_A,
                Q3_B=Q3_B,
                Q3_C=Q3_C,
                args=args,
                image_name=grid_images[i],
            )
            outputs_1.append(output_1)
            outputs_2.append(output_2)
            outputs_3.append(output_3)
            scores_tmp.append(score_tmp)

        has_bad_reply_flag = any(not isinstance(score, int) for score in scores_tmp)
        if has_bad_reply_flag:
            score_avg = "bad reply"
        else:
            score_avg = sum(scores_tmp) / len(scores_tmp)

        write_to_csv(
            csv_path,
            benchmark_name="object_interaction",
            grid_image_name=grid_image_name,
            this_prompt=this_prompt,
            outputs_1=outputs_1,
            outputs_2=outputs_2,
            outputs_3=outputs_3,
            scores_tmp=scores_tmp,
            score_avg=score_avg,
        )

    return csv_path


def model_score(csv_path):
    with open(csv_path, "r") as file:
        reader = csv.reader(file)
        lines = list(reader)
        score = 0
        cnt = 0
        for line in lines[1:]:
            try:
                score_tmp = (float(line[-1]) - 1) / 9
                score += score_tmp
                cnt += 1
            except:
                continue

        score = score / cnt
        print(
            "number of images evaluated: ",
            cnt,
            " object interactions model score: ",
            score,
        )

    with open(csv_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["score: ", score])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default="Qwen/Qwen3-VL-32B-Instruct"
    )
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
        default="playground/results/csv_interaction",
        help="path to store the video scores",
    )
    parser.add_argument(
        "--read-prompt-file",
        type=str,
        default="playground/meta_data/object_interactions.json",
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
