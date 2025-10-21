import os
import json
import random

import csv
import numpy as np
import torch
import matplotlib.pyplot as plt


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
            elif benchmark_name == "background" or benchmark_name == "foreground":
                csv_writer.writerow(
                    [
                        "id",
                        "prompt",
                        "object_1",
                        "d_1",
                        "object_2",
                        "d_2",
                        "mask_name",
                        "xy_json",
                        "change_in_x",
                        "change_in_y",
                    ]
                )
            elif benchmark_name == "2dframe" or benchmark_name == "3dframe":
                csv_writer.writerow(
                    [
                        "video_name",
                        "image_name",
                        "prompt",
                        "object_1",
                        "object_2",
                        "score",
                    ]
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
        elif benchmark_name == "background" or benchmark_name == "foreground":
            csv_writer.writerow(
                [
                    kwargs["vid"],
                    kwargs["prompt"],
                    kwargs["object_1"],
                    kwargs["d_1"],
                    kwargs["object_2"],
                    kwargs["d_2"],
                    kwargs["mask_name"],
                    kwargs["xy_json"],
                    kwargs["change_in_x"],
                    kwargs["change_in_y"],
                ]
            )
        elif benchmark_name == "2dframe" or benchmark_name == "3dframe":
            csv_writer.writerow(
                [
                    kwargs["video_name"],
                    kwargs["image_name"],
                    kwargs["prompt"],
                    kwargs["m0"],
                    kwargs["m1"],
                    kwargs["score_1"],
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
        writer.writerow(["score: ", score_avg])


def combine_frame_spatial_relationships(input_csv, output_csv):
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
                my_score_1 = float(line[-1])

                if my_score_1 < -1:  # =-2
                    my_score_1 = 0
                elif my_score_1 < 0:  # =-1
                    my_score_1 = 0.2
                elif my_score_1 == 0:  # =0
                    my_score_1 = 0.4
                elif my_score_1 > 0:
                    my_score_1 = (my_score_1 * 0.6) + 0.4
                frame_score_1.append(my_score_1)

            score_vid_1.append(sum(frame_score_1) / 16)
            score_frame_1.append(frame_score_1)

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

    return output_csv


def combine_csv_and_cal_model_score(csv_2d, csv_3d, output_file):

    with open(csv_2d, "r") as file:
        reader = csv.reader(file)
        lines_2d = list(reader)
    with open(csv_3d, "r") as file2:
        reader2 = csv.reader(file2)
        lines_3d = list(reader2)

    lines = lines_2d[1:] + lines_3d[1:]
    lines = sorted(lines, key=lambda x: int(x[0]))  # Sort by the first element
    lines = [["id", "score_frame", "Score"]] + lines
    score = []
    print(lines)
    for line in lines[1:]:
        score.append(float(line[-1]))

    score = sum(score) / len(score)

    with open(output_file, "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(lines)
        writer.writerow(["Score: ", score])


def save_mask_data(output_dir, mask_list, box_list, label_list):
    # value = 0  # 0 for background
    value = 1

    mask_img = torch.ones(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        # mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        mask_img[mask.cpu().numpy()[0] == True] = value - 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy(), cmap="gray")
    plt.axis("off")
    plt.savefig(
        os.path.join(output_dir, "mask_background.jpg"),
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.0,
    )

    json_data = [{"value": value, "label": "background"}]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split("(")
        logit = logit[:-1]  # the last is ')'
        json_data.append(
            {
                "value": value,
                "label": name,
                "logit": float(logit),
                "box": box.numpy().tolist(),
            }
        )
    with open(os.path.join(output_dir, "mask.json"), "w") as f:
        json.dump(json_data, f)


def save_mask_foreground(output_dir, mask, obj_prompt):
    # value = 0  # 0 for background
    value = 0

    mask_img = torch.zeros(mask.shape[-2:])
    mask_img[mask.cpu().numpy()[0] == True] = value + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy(), cmap="gray")
    plt.axis("off")
    plt.savefig(
        os.path.join(output_dir, f"mask_foreground_{obj_prompt}.jpg"),
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.0,
    )
