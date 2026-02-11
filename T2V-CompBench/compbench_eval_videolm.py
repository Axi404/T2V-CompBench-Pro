#!/usr/bin/env python3
"""
compbench_eval_videolm.py — Video-LLM evaluation for T2V-CompBench-Pro.

Video-LLM evaluation scripts using Qwen-VL.
Directly processes video input instead of converting to image grids.

Covers all 4 MLLM categories:
  consistent_attr | dynamic_attr | action_binding | interaction

Prerequisites:
  pip install "transformers>=4.57.0" accelerate
  pip install flash-attn --no-build-isolation   # optional, recommended

Recommended models (single 80GB A800):
  Qwen/Qwen3-VL-32B-Instruct          (~66 GB, BF16, best quality)
  Qwen/Qwen3-VL-32B-Instruct-FP8      (~33 GB, FP8, near-lossless)
  Qwen/Qwen3-VL-8B-Instruct           (~16 GB, budget option)

Usage:
  python T2V-CompBench/compbench_eval_videolm.py \\
      --category consistent_attr \\
      --video-path playground/model_output/<model>/consistent_attr_1 \\
      --t2v-model <model_name>

  # Or run all 4 MLLM categories at once:
  python T2V-CompBench/compbench_eval_videolm.py \\
      --category all \\
      --video-path playground/model_output/<model> \\
      --t2v-model <model_name>
"""

import argparse
import csv
import json
import os
import re
import sys
import torch
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.prompt_utils import (
    # Consistent Attr
    CONSISTENT_ATTR_PROMPT_TEMPLATE_Q2 as CA_Q2_TEMPLATE,
    CONSISTENT_ATTR_PROMPT_TEMPLATE_Q3 as CA_Q3_TEMPLATE,
    # Dynamic Attr
    DYNAMIC_ATTR_PROMPT_TEMPLATE_Q1 as DA_Q1_TEMPLATE,
    DYNAMIC_ATTR_PROMPT_TEMPLATE_Q2 as DA_Q2_TEMPLATE,
    DYNAMIC_ATTR_PROMPT_TEMPLATE_Q3 as DA_Q3_TEMPLATE,
    # Action Binding
    ACTION_BINDING_PROMPT_TEMPLATE_Q2 as AB_Q2_TEMPLATE,
    ACTION_BINDING_PROMPT_TEMPLATE_Q3_A as AB_Q3_A_TEMPLATE,
    ACTION_BINDING_PROMPT_TEMPLATE_Q3_BC_OBJ1 as AB_Q3_B_TEMPLATE,
    ACTION_BINDING_PROMPT_TEMPLATE_Q3_BC_OBJ2 as AB_Q3_C_TEMPLATE,
    # Interaction
    INTERACTION_PROMPT_TEMPLATE_Q2 as IN_Q2_TEMPLATE,
    INTERACTION_PROMPT_TEMPLATE_Q3_A as IN_Q3_A_TEMPLATE,
    INTERACTION_PROMPT_TEMPLATE_Q3_B as IN_Q3_B_TEMPLATE,
    INTERACTION_PROMPT_TEMPLATE_Q3_C as IN_Q3_C_TEMPLATE,
)
from utils.utils import extract_json, initialize_csv, set_seed, write_to_csv
from utils.video_utils import read_video_path, convert_video_to_frames

# ── Adapted Q1 prompts for direct video input ──────────────────────────
# Original prompts reference "image grids"; these reference "video" directly.

VIDEO_Q1_CONSISTENT_ATTR = (
    "The following is an AI generated video. "
    "Describe the video, carefully examining objects rendering quality "
    "throughout the video and their visual attributes."
)

VIDEO_Q1_ACTION_BINDING = (
    "The following is an AI generated video. "
    "Describe the video, highlight all the characters and objects "
    "that appear throughout the video and indicate how they act."
)

VIDEO_Q1_INTERACTION = (
    "The following is an AI generated video. "
    "Describe the video, focusing on the interactions between the "
    "characters or objects that appear throughout the video."
)

# Category → (output_path, init_csv_name, write_csv_name, prompt_file, video_subdir)
CATEGORY_CONFIG = {
    "consistent_attr": {
        "output_path": "playground/results/csv_consistent_attr",
        "init_name": "consistent_attr_score",
        "write_name": "consistent_attr",
        "prompt_file": "playground/meta_data/consistent_attribute_binding.json",
        "video_subdir": "consistent_attr_1",
    },
    "dynamic_attr": {
        "output_path": "playground/results/csv_dynamic_attr",
        "init_name": "dynamic_attr_score",
        "write_name": "dynamic_attr",
        "prompt_file": "playground/meta_data/dynamic_attribute_binding.json",
        "video_subdir": "dynamic_attr_2",
    },
    "action_binding": {
        "output_path": "playground/results/csv_action_binding",
        "init_name": "action_binding_score",
        "write_name": "action_binding",
        "prompt_file": "playground/meta_data/action_binding.json",
        "video_subdir": "action_5",
    },
    "interaction": {
        "output_path": "playground/results/csv_interaction",
        "init_name": "object_interaction_score",
        "write_name": "object_interaction",
        "prompt_file": "playground/meta_data/object_interactions.json",
        "video_subdir": "interaction_6",
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  Model loading & generation
# ═══════════════════════════════════════════════════════════════════════


def load_model(model_path: str):
    """Load a Qwen3-VL model and its processor."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[INFO] Loading model: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)

    # Try flash_attention_2, fall back to sdpa if unavailable
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    except (ValueError, ImportError):
        print("[WARN] flash_attention_2 unavailable, using sdpa")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype="auto",
            attn_implementation="sdpa",
            device_map="auto",
        )

    cls_name = type(model).__name__.lower()
    model_type = getattr(model.config, "model_type", "").lower()
    if "qwen3" not in cls_name and "qwen3" not in model_type:
        raise ValueError(
            f"Only Qwen3-VL models are supported, got class={type(model).__name__}, model_type={model_type}"
        )

    print(f"[INFO] Model loaded. dtype={model.dtype}")
    return model, processor


def generate(
    model,
    processor,
    messages: list,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float | None = None,
) -> str:
    """Run a single generation step using the Qwen3-VL native pipeline."""
    do_sample = temperature > 0
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


# ── Message builders ───────────────────────────────────────────────────


def _video_msg(video_path: str, text: str, num_frames: int = 16):
    """User message with video + text."""
    return {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": os.path.abspath(video_path),
                "nframes": num_frames,
            },
            {"type": "text", "text": text},
        ],
    }


def _image_msg(image_path: str, text: str):
    """User message with image + text."""
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": os.path.abspath(image_path)},
            {"type": "text", "text": text},
        ],
    }


def _text_msg(text: str):
    return {"role": "user", "content": text}


def _asst(text: str):
    return {"role": "assistant", "content": text}


# ═══════════════════════════════════════════════════════════════════════
#  1. Consistent Attribute
# ═══════════════════════════════════════════════════════════════════════


def run_consistent_attr_conversation(
    model, processor, video_path: str, Q1: str, Q2: str, Q3: str, args
) -> tuple[str, str, str, int | str]:
    """Single-seed consistent attribute multi-turn conversation."""
    kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
              top_p=args.top_p)

    # Turn 1: describe the video
    msgs = [_video_msg(video_path, Q1, args.num_frames)]
    output_1 = generate(model, processor, msgs, **kw)

    # Turn 2: evaluate phrase_1 (independent follow-up)
    msgs_q2 = [_video_msg(video_path, Q1, args.num_frames),
                _asst(output_1), _text_msg(Q2)]
    output_2 = generate(model, processor, msgs_q2, **kw)

    # Turn 3: evaluate phrase_2 (independent follow-up, replaces Q2)
    msgs_q3 = [_video_msg(video_path, Q1, args.num_frames),
                _asst(output_1), _text_msg(Q3)]
    output_3 = generate(model, processor, msgs_q3, **kw)

    # Parse scores
    try:
        json_obj_2 = extract_json(output_2)
        json_obj_3 = extract_json(output_3)
        option_value_2 = json_obj_2["adjust"]
        option_value_3 = json_obj_3["adjust"]
        option_value = f"{option_value_2}1,{option_value_3}2"
    except Exception:
        option_value = "bad reply"

    score_map = {
        frozenset({"A1,A2", "A2,A1"}): 15,
        frozenset({"A1,B2", "B1,A2", "A2,B1", "B2,A1"}): 14,
        frozenset({"B1,B2", "B2,B1"}): 13,
        frozenset({"A1,C2", "C1,A2", "A2,C1", "C2,A1"}): 12,
        frozenset({"A1,D2", "D1,A2", "A2,D1", "D2,A1"}): 10,
        frozenset({"A1,E2", "E1,A2", "A2,E1", "E2,A1"}): 8,
        frozenset({"B1,C2", "C1,B2", "B2,C1", "C2,B1"}): 11,
        frozenset({"B1,D2", "D1,B2", "B2,D1", "D2,B1"}): 9,
        frozenset({"B1,E2", "E1,B2", "B2,E1", "E2,B1"}): 7,
        frozenset({"C1,C2", "C2,C1"}): 6,
        frozenset({"C1,D2", "D1,C2", "C2,D1", "D2,C1"}): 5,
        frozenset({"C1,E2", "E1,C2", "C2,E1", "E2,C1"}): 3,
        frozenset({"D1,D2", "D2,D1"}): 4,
        frozenset({"D1,E2", "E1,D2", "D2,E1", "E2,D1"}): 2,
        frozenset({"E1,E2", "E2,E1"}): 1,
    }

    score_tmp = "bad reply"
    for keys, val in score_map.items():
        if option_value in keys:
            score_tmp = val
            break
    if score_tmp == "bad reply":
        print("reply wrong format")

    return output_1, output_2, output_3, score_tmp


def eval_consistent_attr(model, processor, args):
    video_list, video_dir = read_video_path(args.video_path)
    with open(args.read_prompt_file, "r") as f:
        prompts = json.load(f)

    csv_path, line_count = initialize_csv(
        args.output_path, args.t2v_model, "consistent_attr_score"
    )
    evaluated = max(line_count - 1, 0)
    print(f"[consistent_attr] {len(video_list)} videos, {evaluated} already done")

    for i in tqdm(range(evaluated, len(video_list)), desc="consistent_attr"):
        video_name = video_list[i]
        video_file = os.path.join(video_dir, video_name)
        num = int(video_name.split(".")[0]) - 1

        phrases = prompts[num]["phrases"]
        this_prompt = prompts[num]["prompt"]
        phrase_1 = phrases.split(";")[0].strip()
        phrase_2 = phrases.split(";")[1].strip()

        Q1 = VIDEO_Q1_CONSISTENT_ATTR
        Q2 = CA_Q2_TEMPLATE.format(phrase_1=phrase_1)
        Q3 = CA_Q3_TEMPLATE.format(phrase_2=phrase_2)

        outputs_1, outputs_2, outputs_3, scores_tmp = [], [], [], []
        for iteration in range(3):
            set_seed(args.seed + iteration)
            o1, o2, o3, s = run_consistent_attr_conversation(
                model, processor, video_file, Q1, Q2, Q3, args
            )
            outputs_1.append(o1)
            outputs_2.append(o2)
            outputs_3.append(o3)
            scores_tmp.append(s)
            print(f"  [{iteration}] {video_name}: {s}")

        if any(not isinstance(s, int) for s in scores_tmp):
            score_avg = "bad reply"
        else:
            score_avg = sum(scores_tmp) / len(scores_tmp)

        write_to_csv(
            csv_path,
            benchmark_name="consistent_attr",
            grid_image_name=video_name,
            this_prompt=this_prompt,
            outputs_1=outputs_1,
            outputs_2=outputs_2,
            outputs_3=outputs_3,
            scores_tmp=scores_tmp,
            score_avg=score_avg,
        )

    return csv_path


def score_consistent_attr(csv_path):
    with open(csv_path, "r") as f:
        lines = list(csv.reader(f))
    score, cnt = 0, 0
    for line in lines[1:]:
        try:
            score += (float(line[-1]) - 1) / 14
            cnt += 1
        except Exception:
            continue
    score = score / cnt
    print(f"[consistent_attr] n={cnt}  score={score:.4f}")
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(["score: ", score])


# ═══════════════════════════════════════════════════════════════════════
#  2. Dynamic Attribute (frame-based, preserves original scoring)
# ═══════════════════════════════════════════════════════════════════════


def _run_da_boundary(model, processor, frame_path, Q1, state_0, state_1, args):
    """Evaluate boundary frame against both states. Returns scores for state_0 and state_1."""
    kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
              top_p=args.top_p)

    msgs_q1 = [_image_msg(frame_path, Q1)]
    output_1 = generate(model, processor, msgs_q1, **kw)

    scores = []
    outputs_2 = []
    for state in [state_0, state_1]:
        Q2 = DA_Q2_TEMPLATE.format(question_group_tmp=state)
        msgs_q2 = [_image_msg(frame_path, Q1), _asst(output_1), _text_msg(Q2)]
        output_2 = generate(model, processor, msgs_q2, **kw)
        outputs_2.append(output_2)

        pattern = r'"score":\s*"([A-D])"'
        match = re.search(pattern, output_2)
        if match:
            score_letter = match.group(1)
        else:
            score_letter = "bad reply"
        score_val = {"A": 1.0, "B": 0.8, "C": 0.2, "D": 0.0}.get(
            score_letter, "bad reply"
        )
        scores.append(score_val)

    return output_1, outputs_2[0], outputs_2[1], scores[0], scores[1]


def _run_da_intermediate(model, processor, frame_path, Q1, Q3, args):
    """Evaluate intermediate frame for state transition."""
    kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
              top_p=args.top_p)

    msgs_q1 = [_image_msg(frame_path, Q1)]
    output_1 = generate(model, processor, msgs_q1, **kw)

    msgs_q3 = [_image_msg(frame_path, Q1), _asst(output_1), _text_msg(Q3)]
    output_3 = generate(model, processor, msgs_q3, **kw)

    pattern = r'"score":\s*(\d+(\.\d+)?),'
    match = re.search(pattern, output_3)
    score_tmp = float(match.group(1)) if match else -1

    return output_1, output_3, score_tmp


def eval_dynamic_attr(model, processor, args):
    # Extract frames (same as original: 8 frames per video)
    frame_folder = args.frame_folder
    if frame_folder is None:
        frame_folder = convert_video_to_frames(args.video_path, num_frames=8)

    with open(args.read_prompt_file, "r") as f:
        prompts = json.load(f)

    csv_path, line_count = initialize_csv(
        args.output_path, args.t2v_model, "dynamic_attr_score"
    )
    frame_dirs = sorted(os.listdir(frame_folder))
    evaluated = max(line_count - 1, 0)
    print(f"[dynamic_attr] {len(frame_dirs)} videos, {evaluated} already done")

    Q1 = DA_Q1_TEMPLATE

    for i in tqdm(range(evaluated, len(frame_dirs)), desc="dynamic_attr"):
        out = []
        inter_answers = []
        frame_dir_name = frame_dirs[i]

        num = int(frame_dir_name) - 1
        this_prompt = prompts[num]["prompt"]
        state_0 = prompts[num]["state 0"]
        state_1 = prompts[num]["state 1"]

        frame_dir_path = os.path.join(frame_folder, frame_dir_name)
        frame_files = sorted(os.listdir(frame_dir_path))
        if len(frame_files) == 0:
            print(f"  no frames for {frame_dir_name}")
            continue

        score = []
        # Boundary: initial frame (index 0) and end frame (index -1)
        for j in range(2):
            set_seed(args.seed)
            state_num = 0 if j == 0 else -1
            frame_path = os.path.join(frame_dir_path, frame_files[state_num])

            o1, o2_s0, o2_s1, sc_s0, sc_s1 = _run_da_boundary(
                model, processor, frame_path, Q1, state_0, state_1, args
            )
            out.extend([o1, o2_s0, o2_s1])
            score.extend([sc_s0, sc_s1])

        score_1_0, score_1_1, score_2_0, score_2_1 = score

        try:
            first_frame_score = score_1_0 * (1 - score_1_1)
            last_frame_score = (1 - score_2_0) * score_2_1
            temp_score = first_frame_score * last_frame_score
        except Exception:
            temp_score = "bad reply"

        if temp_score == "bad reply" or temp_score == 0:
            write_to_csv(
                csv_path,
                benchmark_name="dynamic_attr",
                grid_image_name=frame_dir_name,
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

        # Intermediate frames
        matched_frames_cnt = 0
        has_matched = False
        intermediate_frames = 8 - 2

        for inter_state in range(1, 7):
            set_seed(args.seed)
            frame_path = os.path.join(frame_dir_path, frame_files[inter_state])

            Q3 = DA_Q3_TEMPLATE.format(phrase_0=state_0, phrase_1=state_1)
            o1, o3, sc = _run_da_intermediate(
                model, processor, frame_path, Q1, Q3, args
            )
            out.extend([o1, o3])
            inter_answers.append(o3)

            if sc > 0:
                matched_frames_cnt += 1
            if matched_frames_cnt > intermediate_frames * 0.65:
                has_matched = True
                break

        score_total = temp_score if has_matched else 0
        print(f"  {frame_dir_name}: {score_total}")

        write_to_csv(
            csv_path,
            benchmark_name="dynamic_attr",
            grid_image_name=frame_dir_name,
            this_prompt=this_prompt,
            out=out,
            inter_answers=inter_answers,
            score_1_0=score_1_0,
            score_1_1=score_1_1,
            score_2_0=score_2_0,
            score_2_1=score_2_1,
            flag=1 if has_matched else 0,
            score_total=score_total,
        )

    return csv_path


def score_dynamic_attr(csv_path):
    with open(csv_path, "r") as f:
        lines = list(csv.reader(f))
    score, cnt = 0, 0
    for line in lines[1:]:
        try:
            score += float(line[-1])
            cnt += 1
        except Exception:
            continue
    score = score / cnt
    print(f"[dynamic_attr] n={cnt}  score={score:.4f}")
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(["score: ", score])


# ═══════════════════════════════════════════════════════════════════════
#  3. Action Binding
# ═══════════════════════════════════════════════════════════════════════


def run_action_binding_conversation(
    model, processor, video_path: str, Q1, Q2, Q3_A, Q3_B, Q3_C, args
) -> tuple[str, str, str, int | str]:
    """Single-seed action binding multi-turn conversation."""
    kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
              top_p=args.top_p)

    # Turn 1
    msgs = [_video_msg(video_path, Q1, args.num_frames)]
    output_1 = generate(model, processor, msgs, **kw)

    # Turn 2
    msgs += [_asst(output_1), _text_msg(Q2)]
    output_2 = generate(model, processor, msgs, **kw)

    # Parse Q2 response
    json_obj = extract_json(output_2)
    try:
        option_2 = json_obj["adjust"]
    except Exception:
        option_2 = "bad reply 1"

    if option_2 == "A":
        Q3 = Q3_A
    elif option_2 == "B":
        Q3 = Q3_B
    elif option_2 == "C":
        Q3 = Q3_C
    elif option_2 == "D":
        return output_1, output_2, "", 1
    else:
        return output_1, output_2, "", "bad reply"

    # Turn 3
    msgs += [_asst(output_2), _text_msg(Q3)]
    output_3 = generate(model, processor, msgs, **kw)

    # Parse Q3 response
    adjust_values = []
    for line in output_3.splitlines():
        if '"adjust":' in line:
            value = line.split(":")[1].strip().strip('",')
            adjust_values.append(value)
    option_3 = ",".join(adjust_values)

    # Score based on Q2 + Q3 options
    score_tmp = "bad reply ?"
    if option_3 in ("A1,A2", "A2,A1"):
        score_tmp = 10
    elif option_3 in ("A1,B2", "B1,A2", "A2,B1", "B2,A1"):
        score_tmp = 9
    elif option_3 in ("A1,C2", "C1,A2", "A2,C1", "C2,A1"):
        score_tmp = 8
    elif option_3 in ("B1,B2", "B2,B1"):
        score_tmp = 7
    elif option_3 in ("B1,C2", "C1,B2", "B2,C1", "C2,B1"):
        score_tmp = 6
    elif option_3 in ("C1,C2", "C2,C1"):
        score_tmp = 5
    elif option_3 == "A":
        score_tmp = 4
    elif option_3 == "B":
        score_tmp = 3
    elif option_3 == "C":
        score_tmp = 2
    else:
        print("reply wrong format")

    return output_1, output_2, output_3, score_tmp


def eval_action_binding(model, processor, args):
    video_list, video_dir = read_video_path(args.video_path)
    with open(args.read_prompt_file, "r") as f:
        prompts = json.load(f)

    csv_path, line_count = initialize_csv(
        args.output_path, args.t2v_model, "action_binding_score"
    )
    evaluated = max(line_count - 1, 0)
    print(f"[action_binding] {len(video_list)} videos, {evaluated} already done")

    for i in tqdm(range(evaluated, len(video_list)), desc="action_binding"):
        video_name = video_list[i]
        video_file = os.path.join(video_dir, video_name)
        num = int(video_name.split(".")[0]) - 1

        this_prompt = prompts[num]["prompt"]
        phrase_0 = prompts[num]["phrase_0"]
        phrase_1 = prompts[num]["phrase_1"]

        obj1 = phrase_0[0].split("?")[0]
        obj1_action = phrase_0[1].split("?")[0]
        obj2 = phrase_1[0].split("?")[0]
        obj2_action = phrase_1[1].split("?")[0]

        Q1 = VIDEO_Q1_ACTION_BINDING
        Q2 = AB_Q2_TEMPLATE.format(this_prompt=this_prompt, obj1=obj1, obj2=obj2)
        Q3_A = AB_Q3_A_TEMPLATE.format(
            obj1_action=obj1_action, obj2_action=obj2_action
        )
        Q3_B = AB_Q3_B_TEMPLATE.format(obj1_action=obj1_action)
        Q3_C = AB_Q3_C_TEMPLATE.format(obj2_action=obj2_action)

        outputs_1, outputs_2, outputs_3, scores_tmp = [], [], [], []
        for iteration in range(3):
            set_seed(args.seed + iteration)
            o1, o2, o3, s = run_action_binding_conversation(
                model, processor, video_file, Q1, Q2, Q3_A, Q3_B, Q3_C, args
            )
            outputs_1.append(o1)
            outputs_2.append(o2)
            outputs_3.append(o3)
            scores_tmp.append(s)
            print(f"  [{iteration}] {video_name}: {s}")

        if any(not isinstance(s, int) for s in scores_tmp):
            score_avg = "bad reply"
        else:
            score_avg = sum(scores_tmp) / len(scores_tmp)

        write_to_csv(
            csv_path,
            benchmark_name="action_binding",
            grid_image_name=video_name,
            this_prompt=this_prompt,
            outputs_1=outputs_1,
            outputs_2=outputs_2,
            outputs_3=outputs_3,
            scores_tmp=scores_tmp,
            score_avg=score_avg,
        )

    return csv_path


def score_action_binding(csv_path):
    with open(csv_path, "r") as f:
        lines = list(csv.reader(f))
    score, cnt = 0, 0
    for line in lines[1:]:
        try:
            score += (float(line[-1]) - 1) / 9
            cnt += 1
        except Exception:
            continue
    score = score / cnt
    print(f"[action_binding] n={cnt}  score={score:.4f}")
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(["score: ", score])


# ═══════════════════════════════════════════════════════════════════════
#  4. Interaction
# ═══════════════════════════════════════════════════════════════════════


def run_interaction_conversation(
    model, processor, video_path: str, Q1, Q2, Q3_A, Q3_B, Q3_C, args
) -> tuple[str, str, str, int | str]:
    """Single-seed interaction multi-turn conversation."""
    kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
              top_p=args.top_p)

    # Turn 1
    msgs = [_video_msg(video_path, Q1, args.num_frames)]
    output_1 = generate(model, processor, msgs, **kw)

    # Turn 2
    msgs += [_asst(output_1), _text_msg(Q2)]
    output_2 = generate(model, processor, msgs, **kw)

    # Parse Q2
    json_obj = extract_json(output_2)
    try:
        option_2 = json_obj["adjust"]
    except Exception:
        option_2 = "bad reply 1"

    if option_2 == "A":
        Q3 = Q3_A
    elif option_2 == "B":
        Q3 = Q3_B
    elif option_2 == "C":
        Q3 = Q3_C
    elif option_2 == "D":
        return output_1, output_2, "", 1
    else:
        return output_1, output_2, "", "bad reply"

    # Turn 3
    msgs += [_asst(output_2), _text_msg(Q3)]
    output_3 = generate(model, processor, msgs, **kw)

    # Parse Q3
    json_obj_3 = extract_json(output_3)
    try:
        option_3 = json_obj_3["adjust"]
    except Exception:
        option_3 = "bad reply 2"

    # Score
    score_tmp = "bad reply ?"
    if option_2 == "A":
        score_map = {"A": 10, "B": 9, "C": 8, "D": 7, "E": 6}
        score_tmp = score_map.get(option_3, "bad reply ?")
    elif option_2 == "B":
        score_map = {"A": 5, "B": 3}
        score_tmp = score_map.get(option_3, "bad reply ?")
    elif option_2 == "C":
        score_map = {"A": 4, "B": 2}
        score_tmp = score_map.get(option_3, "bad reply ?")

    if score_tmp == "bad reply ?":
        print("reply wrong format")

    return output_1, output_2, output_3, score_tmp


def eval_interaction(model, processor, args):
    video_list, video_dir = read_video_path(args.video_path)
    with open(args.read_prompt_file, "r") as f:
        prompts = json.load(f)

    csv_path, line_count = initialize_csv(
        args.output_path, args.t2v_model, "object_interaction_score"
    )
    evaluated = max(line_count - 1, 0)
    print(f"[interaction] {len(video_list)} videos, {evaluated} already done")

    for i in tqdm(range(evaluated, len(video_list)), desc="interaction"):
        video_name = video_list[i]
        video_file = os.path.join(video_dir, video_name)
        num = int(video_name.split(".")[0]) - 1

        this_prompt = prompts[num]["prompt"]

        Q1 = VIDEO_Q1_INTERACTION
        Q2 = IN_Q2_TEMPLATE.format(this_prompt=this_prompt)
        Q3_A = IN_Q3_A_TEMPLATE.format(this_prompt=this_prompt)
        Q3_B = IN_Q3_B_TEMPLATE.format(this_prompt=this_prompt)
        Q3_C = IN_Q3_C_TEMPLATE.format(this_prompt=this_prompt)

        outputs_1, outputs_2, outputs_3, scores_tmp = [], [], [], []
        for iteration in range(3):
            set_seed(args.seed + iteration)
            o1, o2, o3, s = run_interaction_conversation(
                model, processor, video_file, Q1, Q2, Q3_A, Q3_B, Q3_C, args
            )
            outputs_1.append(o1)
            outputs_2.append(o2)
            outputs_3.append(o3)
            scores_tmp.append(s)
            print(f"  [{iteration}] {video_name}: {s}")

        if any(not isinstance(s, int) for s in scores_tmp):
            score_avg = "bad reply"
        else:
            score_avg = sum(scores_tmp) / len(scores_tmp)

        write_to_csv(
            csv_path,
            benchmark_name="object_interaction",
            grid_image_name=video_name,
            this_prompt=this_prompt,
            outputs_1=outputs_1,
            outputs_2=outputs_2,
            outputs_3=outputs_3,
            scores_tmp=scores_tmp,
            score_avg=score_avg,
        )

    return csv_path


def score_interaction(csv_path):
    with open(csv_path, "r") as f:
        lines = list(csv.reader(f))
    score, cnt = 0, 0
    for line in lines[1:]:
        try:
            score += (float(line[-1]) - 1) / 9
            cnt += 1
        except Exception:
            continue
    score = score / cnt
    print(f"[interaction] n={cnt}  score={score:.4f}")
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(["score: ", score])


# ═══════════════════════════════════════════════════════════════════════
#  Dispatch
# ═══════════════════════════════════════════════════════════════════════

EVAL_FUNCS = {
    "consistent_attr": (eval_consistent_attr, score_consistent_attr),
    "dynamic_attr": (eval_dynamic_attr, score_dynamic_attr),
    "action_binding": (eval_action_binding, score_action_binding),
    "interaction": (eval_interaction, score_interaction),
}


def run_category(category: str, model, processor, args):
    """Run evaluation + scoring for a single category."""
    eval_fn, score_fn = EVAL_FUNCS[category]

    cfg = CATEGORY_CONFIG[category]
    args.output_path = cfg["output_path"]
    args.read_prompt_file = cfg["prompt_file"]

    csv_path = eval_fn(model, processor, args)
    score_fn(csv_path)
    return csv_path


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video-LLM evaluation for T2V-CompBench-Pro"
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        choices=["consistent_attr", "dynamic_attr", "action_binding", "interaction", "all"],
        help="which MLLM category to evaluate (or 'all')",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-VL-32B-Instruct",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="path to video directory (or base dir if --category all)",
    )
    parser.add_argument("--t2v-model", type=str, required=True, help="model name for CSV")
    parser.add_argument("--num-frames", type=int, default=16,
                        help="frames to sample from each video (for video-based categories)")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)

    # Dynamic attr specific
    parser.add_argument("--frame_folder", type=str, default=None,
                        help="pre-extracted frame folder for dynamic_attr")

    # These are set automatically per category; only used when running a single category
    # with custom paths
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--read-prompt-file", type=str, default=None)

    args = parser.parse_args()

    model, processor = load_model(args.model_path)

    if args.category == "all":
        base = args.video_path
        for cat in ["consistent_attr", "dynamic_attr", "action_binding", "interaction"]:
            subdir = CATEGORY_CONFIG[cat]["video_subdir"]
            args.video_path = os.path.join(base, subdir)
            print(f"\n{'='*60}\n  Evaluating: {cat}\n  Video path: {args.video_path}\n{'='*60}")
            run_category(cat, model, processor, args)
    else:
        if args.output_path is None or args.read_prompt_file is None:
            run_category(args.category, model, processor, args)
        else:
            # User supplied custom paths — run directly
            eval_fn, score_fn = EVAL_FUNCS[args.category]
            csv_path = eval_fn(model, processor, args)
            score_fn(csv_path)
