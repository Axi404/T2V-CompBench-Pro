import argparse
import json
import os
import re
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
    DYNAMIC_ATTR_PROMPT_TEMPLATE_Q1 as Q1_template,
    DYNAMIC_ATTR_PROMPT_TEMPLATE_Q2 as Q2_template,
    DYNAMIC_ATTR_PROMPT_TEMPLATE_Q3 as Q3_template,
)
from utils.utils import set_seed, initialize_csv, write_to_csv
from utils.video_utils import convert_video_to_frames


def run_dynamic_attr_boundary_conversation(
    model,
    processor,
    frame_path: str,
    Q1: str,
    state_0: str,
    state_1: str,
    args,
) -> tuple[str, str, str, float | str, float | str]:
    """
    Run conversation for boundary frames (initial or end frame).

    Args:
        model: The Qwen3-VL model
        processor: The processor
        frame_path: Input frame path
        Q1: First question prompt
        state_0: Initial state description
        state_1: End state description
        args: Arguments containing temperature, top_p, num_beams, max_new_tokens

    Returns:
        tuple: (output_1, output_2_state0, output_2_state1, score_state0, score_state1)
    """
    kw = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
    )
    output_1 = generate_with_messages(
        model, processor, [image_message(frame_path, Q1)], **kw
    )

    outputs_2 = []
    scores = []

    for state in [state_0, state_1]:
        Q2 = Q2_template.format(question_group_tmp=state)
        output_2 = generate_with_messages(
            model,
            processor,
            [image_message(frame_path, Q1), assistant_message(output_1), text_message(Q2)],
            **kw,
        )
        outputs_2.append(output_2)

        # get score from outputs
        pattern = r'"score":\s*"([A-D])"'
        match = re.search(pattern, output_2)
        if match:
            score_tmp = match.group(1)
        else:
            score_tmp = "bad reply"
            print("No score found")
        if score_tmp == "A":
            score_tmp = 1.0
        elif score_tmp == "B":
            score_tmp = 0.8
        elif score_tmp == "C":
            score_tmp = 0.2
        elif score_tmp == "D":
            score_tmp = 0.0
        scores.append(score_tmp)

    return output_1, outputs_2[0], outputs_2[1], scores[0], scores[1]


def run_dynamic_attr_intermediate_conversation(
    model,
    processor,
    frame_path: str,
    Q1: str,
    Q3: str,
    args,
) -> tuple[str, str, float]:
    """
    Run conversation for intermediate frames.

    Args:
        model: The Qwen3-VL model
        processor: The processor
        frame_path: Input frame path
        Q1: First question prompt
        Q3: Third question prompt for intermediate state
        args: Arguments containing temperature, top_p, num_beams, max_new_tokens

    Returns:
        tuple: (output_1, output_3, score)
    """
    kw = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
    )
    output_1 = generate_with_messages(
        model, processor, [image_message(frame_path, Q1)], **kw
    )
    output_3 = generate_with_messages(
        model,
        processor,
        [image_message(frame_path, Q1), assistant_message(output_1), text_message(Q3)],
        **kw,
    )

    # get score from outputs
    pattern = r'"score":\s*(\d+(\.\d+)?),'
    match = re.search(pattern, output_3)
    if match:
        score_tmp = float(match.group(1))
    else:
        score_tmp = -1
        print("No score found")

    return output_1, output_3, score_tmp


def eval_model(args):
    frame_folder = args.frame_folder
    if frame_folder == None:
        video_path = args.video_path
        frame_folder = convert_video_to_frames(video_path, num_frames=8)

    # Model
    if args.model_base is not None or args.conv_mode is not None or args.sep != ",":
        print("[WARN] --model-base/--conv-mode/--sep are deprecated and ignored.")
    model, processor = load_qwen3_model(args.model_path)
    with open(args.read_prompt_file, "r") as json_data:
        prompts = json.load(json_data)

    csv_path, line_count = initialize_csv(
        args.output_path, args.t2v_model, "dynamic_attr_score"
    )

    frame_images = os.listdir(frame_folder)
    frame_images = sorted(frame_images)
    print("[INFO] number of images: ", len(frame_images))

    evaluated = max(line_count - 1, 0)

    for i in range(evaluated, len(frame_images)):
        out = []
        inter_answers = []
        frame_image_name = frame_images[i]
        score = []
        score_total = 0

        num = int(frame_image_name) - 1
        this_prompt = prompts[num]["prompt"]

        state_0 = prompts[num]["state 0"]
        state_1 = prompts[num]["state 1"]
        Q1 = Q1_template

        image_files = os.path.join(frame_folder, frame_images[i])
        image_files = os.listdir(image_files)
        if len(image_files) == 0:
            print(f"no image found for {frame_images[i]}")
            continue
        image_files.sort()

        # conversation for initial and end images
        for j in range(2):
            set_seed(args.seed)
            if j == 0:
                state_num = 0  # get initial image
            else:
                state_num = -1  # get end image
            frame_path = os.path.join(frame_folder, frame_images[i], image_files[state_num])

            # run boundary conversation
            output_1, output_2_s0, output_2_s1, score_s0, score_s1 = (
                run_dynamic_attr_boundary_conversation(
                    model=model,
                    processor=processor,
                    frame_path=frame_path,
                    Q1=Q1,
                    state_0=state_0,
                    state_1=state_1,
                    args=args,
                )
            )
            out.append(output_1)
            out.append(output_2_s0)
            out.append(output_2_s1)
            score.append(score_s0)
            score.append(score_s1)

        score_1_0 = score[0]
        score_1_1 = score[1]
        score_2_0 = score[2]
        score_2_1 = score[3]

        try:
            first_frame_score = score_1_0 * (1 - score_1_1)  # 0~1
            last_frame_score = (1 - score_2_0) * score_2_1  # 0~1
            temp_score = first_frame_score * last_frame_score
        except:
            temp_score = "bad reply"

        if temp_score == "bad reply" or temp_score == 0:
            write_to_csv(
                csv_path,
                benchmark_name="dynamic_attr",
                grid_image_name=frame_image_name,
                this_prompt=this_prompt,
                out=out,
                inter_answers=inter_answers,
                score_1_0=score_1_0,
                score_1_1=score_1_1,
                score_2_0=score_2_0,
                score_2_1=score_2_1,
                flag=-1,
                score_total=0 if temp_score == 0 else "bad reply 4",
            )

        matched_frames_cnt = 0  # for intermediate state
        has_matched_intermediate_frames = False
        intermediate_frames = 8 - 2

        for inter_state in range(1, 7):
            # set seed
            set_seed(args.seed)

            # prepare image input
            frame_path = os.path.join(
                frame_folder, frame_images[i], image_files[inter_state]
            )

            # run intermediate conversation
            Q3 = Q3_template.format(phrase_0=state_0, phrase_1=state_1)
            output_1, output_3, score_tmp = run_dynamic_attr_intermediate_conversation(
                model=model,
                processor=processor,
                frame_path=frame_path,
                Q1=Q1,
                Q3=Q3,
                args=args,
            )
            out.append(output_1)
            out.append(output_3)
            inter_answers.append(output_3)

            if score_tmp > 0:
                matched_frames_cnt += 1

            if matched_frames_cnt > intermediate_frames * 0.65:
                has_matched_intermediate_frames = True
                break
        score_total = temp_score if has_matched_intermediate_frames else 0
        print("score total for", frame_images[i], score_total)

        write_to_csv(
            csv_path,
            benchmark_name="dynamic_attr",
            grid_image_name=frame_image_name,
            this_prompt=this_prompt,
            out=out,
            inter_answers=inter_answers,
            score_1_0=score_1_0,
            score_1_1=score_1_1,
            score_2_0=score_2_0,
            score_2_1=score_2_1,
            flag=1 if has_matched_intermediate_frames else 0,
            score_total=score_total,
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
                score_tmp = float(line[-1])
                score += score_tmp
                cnt += 1
            except:
                continue

        score = score / cnt
        print(
            "number of images evaluated: ",
            cnt,
            " dynamic attribute binding model score: ",
            score,
        )

    with open(csv_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["score: ", score])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-VL-32B-Instruct",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--output-path",
        type=str,
        default="playground/results/csv_dynamic_attr",
        help="path to store the video scores",
    )
    parser.add_argument(
        "--read-prompt-file",
        type=str,
        default="playground/meta_data/dynamic_attribute_binding.json",
        help="path of json file with meta data",
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
        "--frame_folder",
        type=str,
        default=None,
        help="image grid path",
    )
    args = parser.parse_args()

    csv_path = eval_model(args)
    model_score(csv_path)
