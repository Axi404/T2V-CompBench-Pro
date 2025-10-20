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


def initialize_csv(output_path: str, t2v_model: str, benchmark_name: str):
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, f"{t2v_model}_{benchmark_name}.csv")

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
            if benchmark_name == "dynamic_attr":
                csv_writer.writerow(
                    "name",
                    "prompt",
                    "1_answer1",
                    "1_answer2",
                    "2_answer3",
                    "2_answer1",
                    "2_answer2",
                    "2_answer3",
                    "inter_answers",
                    "score_1",
                    "score_1_1",
                    "score_2",
                    "score_2_1",
                    "flag",
                    "Score",
                )
            elif benchmark_name == "numeracy_frame":
                csv_writer.writerow(
                    [
                        "video_name",
                        "image_name",
                        "prompt",
                        "objects",
                        "numbers",
                        "actual_numbers",
                        "score",
                    ]
                )
            else:
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


def write_to_csv(
    csv_path,
    benchmark_name,
    **kwargs,
):
    csvfile = open(csv_path, "a", newline="")
    try:
        csv_writer = csv.writer(csvfile)
        if benchmark_name == "dynamic_attr":
            csv_writer.writerow(
                [
                    kwargs["grid_image_name"],
                    kwargs["this_prompt"],
                    kwargs["out"][0],
                    kwargs["out"][1],
                    kwargs["out"][2],
                    kwargs["out"][3],
                    kwargs["out"][4],
                    kwargs["out"][5],
                    kwargs["inter_answers"],
                    kwargs["score_1_0"],
                    kwargs["score_1_1"],
                    kwargs["score_2_0"],
                    kwargs["score_2_1"],
                    kwargs["flag"],
                    kwargs["score_total"],
                ]
            )
        elif benchmark_name == "numeracy_frame":
            csv_writer.writerow(
                kwargs["video_name"],
                kwargs["image_name"],
                kwargs["prompt"],
                kwargs["objects"],
                kwargs["numbers"],
                kwargs["actual_numbers"],
                kwargs["score"],
            )
        else:
            csv_writer.writerow(
                [
                    kwargs["grid_image_name"],
                    kwargs["this_prompt"],
                    kwargs["outputs_1"][0],
                    kwargs["outputs_1"][1],
                    kwargs["outputs_1"][2],
                    kwargs["scores_tmp"][0],
                    kwargs["outputs_2"][0],
                    kwargs["outputs_2"][1],
                    kwargs["outputs_2"][2],
                    kwargs["scores_tmp"][1],
                    kwargs["outputs_3"][0],
                    kwargs["outputs_3"][1],
                    kwargs["outputs_3"][2],
                    kwargs["scores_tmp"][2],
                    kwargs["score_avg"],
                ]
            )
        csvfile.flush()
    finally:
        csvfile.close()


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
        print("number of images evaluated: ", cnt, " model score: ", score)

    with open(csv_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["score: ", score])

def combine_frame_numeracy(input_csv, output_csv):
    score_total = 0
    cnt = 0
    with open(input_csv, "r") as file:
        reader = csv.reader(file)
        lines = list(reader)

        batch_size = 16
        num_vid = (len(lines) - 1) / batch_size
        if num_vid != int(num_vid):
            print("error: number of lines WRONG")

        score_vid_1 = []
        id = []
        score_frame_1 = []

        for i in range(int(num_vid)):
            batch = lines[i * batch_size + 1 : (i + 1) * batch_size + 1]
            # Process the batch of lines
            frame_score_1 = []

            id.append(batch[0][0])
            for line in batch:
                frame_score_1.append(float(line[-1]))

            score_tmp = sum(frame_score_1) / 16
            score_vid_1.append(score_tmp)
            score_frame_1.append(frame_score_1)
            cnt += 1
            score_total += score_tmp

    score_avg = score_total / cnt

    ####################################
    # socre_obj_1 = score_vid_1[0:20]
    # score_one_obj_2 = score_vid_1[20:50]
    # score_one_obj_3 = score_vid_1[50:80]
    # score_one_obj_4 = score_vid_1[80:110]
    # score_one_obj_5 = score_vid_1[110:140]
    # score_one_obj_678 = score_vid_1[140:160]
    # score_two_obj_low = score_vid_1[160:175]
    # score_two_obj_mid = score_vid_1[175:190]
    # score_two_obj_high = score_vid_1[190:200]

    # socre_obj_1 = sum(socre_obj_1) / len(socre_obj_1)
    # score_one_obj_2 = sum(score_one_obj_2) / len(score_one_obj_2)
    # score_one_obj_3 = sum(score_one_obj_3) / len(score_one_obj_3)
    # score_one_obj_4 = sum(score_one_obj_4) / len(score_one_obj_4)
    # score_one_obj_5 = sum(score_one_obj_5) / len(score_one_obj_5)
    # score_one_obj_678 = sum(score_one_obj_678) / len(score_one_obj_678)
    # score_two_obj_low = sum(score_two_obj_low) / len(score_two_obj_low)
    # score_two_obj_mid = sum(score_two_obj_mid) / len(score_two_obj_mid)
    # score_two_obj_high = sum(score_two_obj_high) / len(score_two_obj_high)
    ####################################

    print("number of videos evaluated: ", cnt, " numeracy model score: ", score_avg)

    score_vid_1 = ["Score_1"] + score_vid_1
    id = ["id"] + id
    score_frame_1 = ["Score_frame_1"] + score_frame_1

    if len(score_vid_1) != len(score_frame_1) != len(id):
        print("counting error")

    with open(output_csv, "w") as output_file:
        writer = csv.writer(output_file)
        for i in range(len(id)):
            # Append data to the end of each row
            row = [id[i], score_frame_1[i], score_vid_1[i]]
            # Write the modified row to the new file
            writer.writerow(row)

    with open(output_csv, "a", newline="") as file:
        writer = csv.writer(file)

        # writer.writerow(["socre_obj_1: ",socre_obj_1])
        # writer.writerow(["score_one_obj_2: ",score_one_obj_2])
        # writer.writerow(["score_one_obj_3: ",score_one_obj_3])
        # writer.writerow(["score_one_obj_4: ",score_one_obj_4])
        # writer.writerow(["score_one_obj_5: ",score_one_obj_5])
        # writer.writerow(["score_one_obj_678: ",score_one_obj_678])
        # writer.writerow(["score_two_obj_low: ",score_two_obj_low])
        # writer.writerow(["score_two_obj_mid: ",score_two_obj_mid])
        # writer.writerow(["score_two_obj_high: ",score_two_obj_high])
        writer.writerow(["score: ", score_avg])