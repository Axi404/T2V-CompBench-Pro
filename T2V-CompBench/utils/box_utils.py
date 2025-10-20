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
