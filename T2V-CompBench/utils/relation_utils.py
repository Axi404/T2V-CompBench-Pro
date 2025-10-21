import torch

def calculate_iou(box0, box1):
    # Calculate IoU
    # from xywh to xyxy
    [box0_xmin, box0_ymin] = box0[:2] - box0[2:] / 2
    [box0_xmax, box0_ymax] = box0[:2] + box0[2:] / 2
    [box1_xmin, box1_ymin] = box1[:2] - box1[2:] / 2
    [box1_xmax, box1_ymax] = box1[:2] + box1[2:] / 2

    x_overlap = max(0, min(box0_xmax, box1_xmax) - max(box0_xmin, box1_xmin))
    y_overlap = max(0, min(box0_ymax, box1_ymax) - max(box0_ymin, box1_ymin))

    intersection = x_overlap * y_overlap
    box0_area = box0[2] * box0[3]
    box1_area = box1[2] * box1[3]
    union = box0_area + box1_area - intersection

    IoU = intersection / union  # intersection over union
    IoMinA = intersection / min(
        box0_area, box1_area
    )  # intersection over the smaller box area
    IoU = IoU.item()
    IoMinA = IoMinA.item()
    return IoU, IoMinA


def filter_box(boxes, phrases, probs, iou_threshold):
    new_boxes = []
    new_probs = []
    new_phrases = []
    for i in range(len(boxes)):
        flag = 0
        for j in range(len(new_boxes)):
            IoU, _ = calculate_iou(boxes[i], new_boxes[j])
            if IoU > iou_threshold:
                flag = 1
                if probs[i] > new_probs[j]:
                    new_boxes[j] = boxes[i]
                    new_probs[j] = probs[i]
                    new_phrases[j] = phrases[i]
                break
        if flag == 0:
            new_boxes.append(boxes[i])
            new_probs.append(probs[i])
            new_phrases.append(phrases[i])

    return new_boxes, new_phrases, new_probs

def spatial_judge(box0, box1, spatial):
    # box:[xc,yc,w,h]
    correct_spatial = False
    IoU = 0
    IoMinA = 0

    IoU, IoMinA = calculate_iou(box0, box1)

    centre_0 = [box0[0].item(), box0[1].item()]
    centre_1 = [box1[0].item(), box1[1].item()]
    dw = centre_0[0] - centre_1[0]  # x0-x1
    dh = centre_0[1] - centre_1[1]  # y0-y1

    if dw < 0.0 and abs(dw) > abs(dh):
        actual_spatial = "left"
    elif dw > 0.0 and abs(dw) > abs(dh):
        actual_spatial = "right"
    elif dh < 0.0 and abs(dh) > abs(dw):
        actual_spatial = "above"
    elif dh > 0.0 and abs(dh) > abs(dw):
        actual_spatial = "under"
    else:
        actual_spatial = ""

    if actual_spatial == spatial:
        correct_spatial = True
    elif spatial == "on" and actual_spatial == "above":
        correct_spatial = True
    elif spatial == "below" and actual_spatial == "under":
        correct_spatial = True
    else:
        correct_spatial = False

    return actual_spatial, correct_spatial, centre_0, centre_1, IoU, IoMinA


def pick_max_2d(total_score_1_list, record_all_correct_spatial):
    max1 = max(total_score_1_list)
    ind1 = total_score_1_list.index(max1)
    best_box = record_all_correct_spatial[ind1]
    score = best_box["spatial_score_1"]
    selected_box_0 = best_box["box0"]
    selected_box_1 = best_box["box1"]
    selected_label = best_box["label"]
    return score, selected_box_0, selected_box_1, selected_label


def intersection_judge(box0, box1):
    # box:[xc,yc,w,h]
    IoU, IoMinA = calculate_iou(box0, box1)
    return IoU, IoMinA


def pick_max_3d(total_score_1_list, record_all_good_spatial):
    max1 = max(total_score_1_list)
    ind1 = total_score_1_list.index(max1)
    best_box = record_all_good_spatial[ind1]
    score = best_box["spatial_score_1"]
    mask0 = best_box["mask0"]
    mask1 = best_box["mask1"]
    return score, mask0, mask1


def clean_boxes(boxes, size):
    H = size[1]
    W = size[0]
    clean_boxes = []
    m = 1
    for i in range(len(boxes)):
        box = boxes[i]
        box = box * torch.Tensor([W, H, W, H])
        clean_boxes.append(box)
    if len(clean_boxes) == 0:
        m = 0
    return clean_boxes, m