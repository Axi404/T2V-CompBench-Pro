from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_boxes_to_image(
    image_pil: Image.Image, tgt: dict, prompt_objs: list, color1=None, color2=None
) -> tuple[Image.Image, Image.Image]:
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    objs = [label.split("(")[0] for label in labels]
    if len(prompt_objs) == 1 and color1 is None:
        color1 = (255, 0, 0)
    elif len(prompt_objs) == 2 and color1 is None and color2 is None:
        color1 = (255, 0, 0)
        color2 = (0, 0, 255)
    elif color1 is None or color2 is None:
        print("wrong objects")

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label, obj in zip(boxes, labels, objs):
        # TODO: spatial relationships
        # if obj == prompt_objs[0]:  # obj1
        #     color = color1
        # elif obj == prompt_objs[1]:  # obj2
        #     color = color2
        # else:
        #     print("wrong objects")
        #     color = (0, 0, 0)
        # Check the difference
        if obj == prompt_objs[0]:  # obj1
            color = color1
        elif len(prompt_objs) == 2:
            if obj == prompt_objs[1]:  # obj2
                color = color2
        else:
            print("wrong objects")
            color = (0, 0, 0)

        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        xc = int(box[0])
        yc = int(box[1])
        s = 3
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]

        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")
        draw.ellipse((xc - s, yc - s, xc + s, yc + s), fill=color)

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)
    return image_pil, mask


def show_mask(
    mask,
    ax,
    my_color=np.array([255 / 255, 255 / 255, 255 / 255, 0.6]),
    random_color=False,
):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = my_color
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )
    ax.text(x0, y0, label)
