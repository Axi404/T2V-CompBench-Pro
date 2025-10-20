import os
import csv
import json
import random
import numpy as np
import torch


def extract_json(string):
    # Find the start and end positions of the JSON part
    start = string.find("{")
    end = string.rfind("}") + 1

    # Extract the JSON part from the string
    json_part = string[start:end]

    # Load the JSON part as a dictionary
    try:
        json_data = json.loads(json_part)
    except json.JSONDecodeError:
        # Handle the case when the JSON part is not valid
        print("Invalid JSON part")
        return None

    return json_data


def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize_csv(output_path: str, t2v_model: str):
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, f"{t2v_model}_action_binding_score.csv")

    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as csvreader:
            reader = csv.reader(csvreader)
            lines = list(reader)  # Read all lines into a list
            line_count = len(lines)  # Count the number of lines
    else:
        line_count = 0

    if line_count == 0:
        csvfile = open(csv_path, "a", newline="")
        try:
            csv_writer = csv.writer(csvfile)
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
        finally:
            csvfile.close()

    return csv_path, line_count


def model_score(csv_path):
    with open(csv_path, "r") as file:
        reader = csv.reader(file)
        lines = list(reader)
        score = 0
        cnt = 0
        for line in lines[1:]:
            try:
                score_tmp = (float(line[-1]) - 1) / 9  # normalize
                score += score_tmp
                cnt += 1
            except:
                continue
        score = score / cnt
        print(
            "number of images evaluated: ", cnt, " action binding model score: ", score
        )

    with open(csv_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["score: ", score])