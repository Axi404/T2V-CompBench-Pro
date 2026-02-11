import argparse
import os
import sys
import csv
import json

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


def evaluate_single_grid(
    model,
    processor,
    image_path: str,
    Q1: str,
    Q2: str,
    Q3: str,
    args,
    grid_image_name: str,
) -> tuple[str, str, str, int | str]:
    """
    Evaluate a single grid image for consistent attribute binding (single iteration).

    Args:
        model: the Qwen3-VL model
        processor: the processor
        image_path: input image path
        Q1: first question template
        Q2: second question template
        Q3: third question template
        args: argument namespace containing temperature, top_p, num_beams, max_new_tokens
        grid_image_name: name of the grid image being evaluated

    Returns:
        output_1: first conversation output
        output_2: second conversation output
        output_3: third conversation output
        score: score for this evaluation
    """
    kw = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
    )

    output_1 = generate_with_messages(
        model, processor, [image_message(image_path, Q1)], **kw
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

    print("--------------------------------")
    print("output_1", output_1)
    print("output_2", output_2)
    print("output_3", output_3)

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
        score = 15
    elif option_value in ["A1,B2", "B1,A2", "A2,B1", "B2,A1"]:
        score = 14
    elif option_value in ["B1,B2", "B2,B1"]:
        score = 13
    elif option_value in ["A1,C2", "C1,A2", "A2,C1", "C2,A1"]:
        score = 12
    elif option_value in ["A1,D2", "D1,A2", "A2,D1", "D2,A1"]:
        score = 10
    elif option_value in ["A1,E2", "E1,A2", "A2,E1", "E2,A1"]:
        score = 8
    elif option_value in ["B1,C2", "C1,B2", "B2,C1", "C2,B1"]:
        score = 11
    elif option_value in ["B1,D2", "D1,B2", "B2,D1", "D2,B1"]:
        score = 9
    elif option_value in ["B1,E2", "E1,B2", "B2,E1", "E2,B1"]:
        score = 7
    elif option_value in ["C1,C2", "C2,C1"]:
        score = 6
    elif option_value in ["C1,D2", "D1,C2", "C2,D1", "D2,C1"]:
        score = 5
    elif option_value in ["C1,E2", "E1,C2", "C2,E1", "E2,C1"]:
        score = 3
    elif option_value in ["D1,D2", "D2,D1"]:
        score = 4
    elif option_value in ["D1,E2", "E1,D2", "D2,E1", "E2,D1"]:
        score = 2
    elif option_value in ["E1,E2", "E2,E1"]:
        score = 1
    else:
        score = "bad reply"
        print("reply wrong format")

    print("score for", grid_image_name, score)

    return output_1, output_2, output_3, score


def eval_model(args):
    # preprocess: video to image grid
    print("Preprocessing video to image grid...")
    image_grid_path = args.image_grid_path
    if image_grid_path == None:
        video_path = args.video_path
        image_grid_path = convert_video_to_grid(video_path, split=5)

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

    # Group grid images by video name (e.g., "0001_0.png", "0001_1.png" -> "0001": [...])
    _grid_images = [f for f in os.listdir(image_grid_path) if f[0].isdigit()]
    _grid_images = sorted(_grid_images)

    grid_images = {}
    for _grid_image in _grid_images:
        # Extract video name (part before the last underscore, e.g., "0001" from "0001_0.png")
        name = (
            _grid_image.rsplit("_", 1)[0]
            if "_" in _grid_image
            else _grid_image.split(".")[0]
        )
        if name not in grid_images:
            grid_images[name] = []
        grid_images[name].append(_grid_image)

    video_names = sorted(grid_images.keys())
    print(f"Total videos to evaluate: {len(video_names)}")

    evaluated = max(line_count - 1, 0)

    for i in range(evaluated, len(video_names)):
        video_name = video_names[i]
        grid_files = grid_images[video_name]
        print(
            f"Evaluating video {i+1}/{len(video_names)}: {video_name} ({len(grid_files)} grids)..."
        )

        # Get prompt info from video name
        num = int(video_name[0:4]) - 1
        phrases = prompts[num]["phrases"]
        this_prompt = prompts[num]["prompt"]
        phrase_1 = phrases.split(";")[0].strip()
        phrase_2 = phrases.split(";")[1].strip()

        Q1 = Q1_template
        Q2 = Q2_template.format(phrase_1=phrase_1)
        Q3 = Q3_template.format(phrase_2=phrase_2)

        # Collect outputs and scores across all grids
        all_outputs_1 = []
        all_outputs_2 = []
        all_outputs_3 = []
        all_scores = []

        # Iterate over each grid segment of the video
        for grid_file in grid_files:
            print(f"  Processing grid: {grid_file}")
            image_path = os.path.join(image_grid_path, grid_file)

            # Run 3 iterations for each grid
            for iteration in range(3):
                set_seed(args.seed + iteration)
                output_1, output_2, output_3, score = evaluate_single_grid(
                    model=model,
                    processor=processor,
                    image_path=image_path,
                    Q1=Q1,
                    Q2=Q2,
                    Q3=Q3,
                    args=args,
                    grid_image_name=grid_file,
                )
                all_outputs_1.append(output_1)
                all_outputs_2.append(output_2)
                all_outputs_3.append(output_3)
                all_scores.append(score)

        # Calculate average score across all grids and iterations
        has_bad_reply_flag = any(not isinstance(s, int) for s in all_scores)
        if has_bad_reply_flag:
            score_avg = "bad reply"
        else:
            score_avg = sum(all_scores) / len(all_scores)

        print(f"  Video {video_name} final score: {score_avg}")

        write_to_csv(
            csv_path,
            benchmark_name="consistent_attr",
            grid_image_name=video_name,
            this_prompt=this_prompt,
            outputs_1=all_outputs_1,
            outputs_2=all_outputs_2,
            outputs_3=all_outputs_3,
            scores_tmp=all_scores,
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
