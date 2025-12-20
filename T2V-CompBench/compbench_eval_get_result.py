import csv
import os
import argparse

def model_score_action_binding(model_name):
    csv_path = f"playground/results/csv_action_binding/{model_name}_action_binding_score.csv"
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        lines = list(reader)
        score = 0
        cnt = 0
        for line in lines[1:]:
            try:
                score_tmp = (float(line[-1])-1)/9   # normalize 
                score+=score_tmp
                cnt+=1
            except:
                continue
        score = score/cnt
    return score


def model_score_dynamic_attr(model_name):
    csv_path = f"playground/results/csv_dynamic_attr/{model_name}_dynamic_attr_score.csv"
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        lines = list(reader)
        score = 0
        cnt = 0
        for line in lines[1:]:
            try:
                score_tmp = float(line[-1]) 
                score+=score_tmp
                cnt+=1
            except:
                continue
        
        score = score/cnt
    return score


def model_score_interaction(model_name):
    csv_path = f"playground/results/csv_interaction/{model_name}_object_interaction_score.csv"
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        lines = list(reader)
        score = 0
        cnt = 0
        for line in lines[1:]:
            try:
                score_tmp = (float(line[-1])-1)/9 
                score+=score_tmp
                cnt+=1
            except:
                continue

        score = score/cnt
    return score

def model_score_consistent_attr(model_name):
    csv_path = f"playground/results/csv_consistent_attr/{model_name}_consistent_attr_score.csv"
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        lines = list(reader)
        score = 0
        cnt = 0
        for line in lines[1:]:
            try:
                score_tmp = (float(line[-1])-1)/14 
                score+=score_tmp
                cnt+=1
            except:
                continue
        
        score = score/cnt
    return score

def model_score_numeracy(model_name):
    csv_path = f"playground/results/csv_numeracy/{model_name}_numeracy_frame.csv"
    score_total = 0
    cnt = 0
    with open(csv_path, "r") as file:
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
    return score_avg


def model_score_spatial_relationships(model_name):
    csv_2d = f"playground/results/csv_spatial/{model_name}_2dvideo.csv"
    csv_3d = f"playground/results/csv_spatial/{model_name}_3dvideo.csv"
    with open(csv_2d, 'r') as file:
        reader = csv.reader(file)
        lines_2d = list(reader)
    with open(csv_3d, 'r') as file2:
        reader2 = csv.reader(file2)
        lines_3d = list(reader2)
    lines = lines_2d[1:] + lines_3d[1:]
    lines = sorted(lines, key=lambda x: int(x[0]))  # Sort by the first element
    lines = [["id","score_frame","Score"]] + lines
    score = []
    for line in lines[1:]:
        score.append(float(line[-1]))
    score = sum(score)/len(score)
    return score



def model_score_motion_binding(model_name):
    csv_path = f"playground/results/csv_motion_binding/{model_name}_motion_score.csv"
    new_csv = []
    with open(csv_path, "r") as file:
        reader = csv.reader(file)
        lines = list(reader)
        score = (
            0  # neither detected: -1, detected: motion score 0~1, total scale: -1 ~ 1
        )
        cnt = 0
        score_pos = 0
        cnt_pos = 0

        new_header = lines[0]
        new_header[-1] = "Score_before_adjust"
        new_header.append("Score")
        new_csv.append(new_header)

        for line in lines[1:]:

            score_tmp = float(line[-1])

            if score_tmp < 0:
                score_tmp = 0
            elif score_tmp >= 0:
                score_pos += score_tmp
                score_tmp = (score_tmp * 0.8) + 0.2
                cnt_pos += 1

            new_line = line
            new_line.append(score_tmp)
            new_csv.append(new_line)

            score += score_tmp

            cnt += 1

        score = score / cnt
        score_pos = score_pos / cnt_pos
    return score
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--t2v-model", type=str, required=True)
    args = args.parse_args()
    action_binding_score = model_score_action_binding(args.t2v_model)
    consistent_attr_score = model_score_consistent_attr(args.t2v_model)
    dynamic_attr_score = model_score_dynamic_attr(args.t2v_model)
    interaction_score = model_score_interaction(args.t2v_model)
    numeracy_score = model_score_numeracy(args.t2v_model)
    spatial_score = model_score_spatial_relationships(args.t2v_model)
    motion_binding_score = model_score_motion_binding(args.t2v_model)
    print(f"Model Name: {args.t2v_model}")
    print(f"Action Binding Score: {action_binding_score:.4f}")
    print(f"Consistent Attribute Score: {consistent_attr_score:.4f}")
    print(f"Dynamic Attribute Score: {dynamic_attr_score:.4f}")
    print(f"Interaction Score: {interaction_score:.4f}")
    print(f"Numeracy Score: {numeracy_score:.4f}")
    print(f"Spatial Score: {spatial_score:.4f}")
    print(f"Motion Binding Score: {motion_binding_score:.4f}")
    print(f"Total Score: {(action_binding_score + consistent_attr_score + dynamic_attr_score + interaction_score + numeracy_score + spatial_score + motion_binding_score)/7:.4f}")