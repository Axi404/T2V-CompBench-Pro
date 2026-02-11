import argparse
import os
import sys
import csv
import json
from tqdm import tqdm

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
    CONSISTENT_ATTR_PROMPT_TEMPLATE_Q1 as Q1_template,
    CONSISTENT_ATTR_PROMPT_TEMPLATE_Q2 as Q2_template,
    CONSISTENT_ATTR_PROMPT_TEMPLATE_Q3 as Q3_template,
)
from utils.utils import (
    extract_json,
    initialize_csv,
    set_seed,
    write_to_csv,
)
from utils.video_utils import convert_video_to_grid


def run_consistent_attr_conversation(
    model,
    processor,
    image_path: str,
    Q1: str,
    Q2: str,
    Q3: str,
    args,
    image_name: str = "",
) -> tuple[str, str, str, int | str]:
    """
    Run a single iteration of the consistent attribute multi-turn conversation.

    Args:
        model: The Qwen3-VL model
        processor: The processor
        image_path: Input image path
        Q1: First question prompt
        Q2: Second question prompt (for phrase_1)
        Q3: Third question prompt (for phrase_2)
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

    output_1 = generate_with_messages(
        model,
        processor,
        [image_message(image_path, Q1)],
        **kw,
    )

    output_2 = generate_with_messages(
        model,
        processor,
        [image_message(image_path, Q1), assistant_message(output_1), text_message(Q2)],
        **kw,
    )

    output_3 = generate_with_messages(
        model,
        processor,
        [image_message(image_path, Q1), assistant_message(output_1), text_message(Q3)],
        **kw,
    )

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

    return output_1, output_2, output_3, score_tmp


def eval_model(args):
    # preprocess: video to image grid
    print("Preprocessing video to image grid...")
    image_grid_path = args.image_grid_path
    if image_grid_path == None:
        video_path = args.video_path
        image_grid_path = convert_video_to_grid(video_path)

    # Model
    print("Loading model...")
    if args.model_base is not None or args.conv_mode is not None or args.sep != ",":
        print("[WARN] --model-base/--conv-mode/--sep are deprecated and ignored.")
    model, processor = load_qwen3_model(args.model_path)
    print("Loading prompts...")
    with open(args.read_prompt_file, "r") as json_data:
        prompts = json.load(json_data)

    print("Initializing CSV...")
    csv_path, line_count = initialize_csv(
        args.output_path, args.t2v_model, "consistent_attr_score"
    )

    grid_images = [f for f in os.listdir(image_grid_path) if f[0].isdigit()]
    grid_images = sorted(grid_images)
    print(len(grid_images))

    evaluated = max(line_count - 1, 0)

    for i in tqdm(range(evaluated, len(grid_images)), desc="Evaluating samples"):
        # get image name
        grid_image_name = grid_images[i]
        num = int(grid_image_name[0:4]) - 1

        # prepare text inputs
        phrases = prompts[num]["phrases"]
        this_prompt = prompts[num]["prompt"]
        phrase_1 = phrases.split(";")[0].strip()
        phrase_2 = phrases.split(";")[1].strip()

        image_path = os.path.join(image_grid_path, grid_images[i])

        Q1 = Q1_template
        Q2 = Q2_template.format(phrase_1=phrase_1)
        Q3 = Q3_template.format(phrase_2=phrase_2)

        outputs_1 = []
        outputs_2 = []
        outputs_3 = []
        scores_tmp = []

        for iteration in range(3):
            set_seed(args.seed + iteration)

            # run multi-turn conversation
            output_1, output_2, output_3, score_tmp = run_consistent_attr_conversation(
                model=model,
                processor=processor,
                image_path=image_path,
                Q1=Q1,
                Q2=Q2,
                Q3=Q3,
                args=args,
                image_name=grid_images[i],
            )
            outputs_1.append(output_1)
            outputs_2.append(output_2)
            outputs_3.append(output_3)
            scores_tmp.append(score_tmp)
            print(f"[{iteration}] score for {grid_images[i]}: {score_tmp}")

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
        default="playground/results/csv_consistent_attr",
        help="path to store the video scores",
    )
    parser.add_argument(
        "--read-prompt-file",
        type=str,
        default="playground/meta_data/consistent_attribute_binding.json",
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
