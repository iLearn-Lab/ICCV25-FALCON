from typing import Dict, Any, Tuple, Optional

from PIL import Image
import re


def de_norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h
    box = x1, y1, x2, y2
    return box


def box_xywh_to_xyxy(box, *, w=None, h=None):
    x, y, bw, bh = box
    x2 = x + bw
    y2 = y + bh
    if w is not None:
        x2 = min(x2, w)
    if h is not None:
        y2 = min(y2, h)
    box = x, y, x2, y2
    return box


""" box absolute x,y (int) -> relative x,y (float) 
"""
def norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box

    # Calculate the normalized coordinates with min-max clamping
    norm_x1 = max(0.0, min(x1 / w, 1.0))
    norm_y1 = max(0.0, min(y1 / h, 1.0))
    norm_x2 = max(0.0, min(x2 / w, 1.0))
    norm_y2 = max(0.0, min(y2 / h, 1.0))

    # Return the normalized box coordinates
    normalized_box = (round(norm_x1, 3), round(norm_y1, 3), round(norm_x2, 3), round(norm_y2, 3))
    return normalized_box


""" point absolute x,y (int) -> relative x,y (float) 
"""
def norm_point_xyxy(point, *, w, h):
    x, y = point
    norm_x = max(0.0, min(x / w, 1.0))
    norm_y = max(0.0, min(y / h, 1.0))
    point = norm_x, norm_y
    return point


def expand2square(pil_img, background_color=(255, 255, 255)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def box_xyxy_expand2square(box, *, w, h):
    if w == h:
        return box
    if w > h:
        x1, y1, x2, y2 = box
        y1 += (w - h) // 2
        y2 += (w - h) // 2
        box = x1, y1, x2, y2
        return box
    assert w < h
    x1, y1, x2, y2 = box
    x1 += (h - w) // 2
    x2 += (h - w) // 2
    box = x1, y1, x2, y2
    return box


def point_xy_expand2square(point, *, w, h):
    pseudo_box = (point[0], point[1], point[0], point[1])
    expanded_box = box_xyxy_expand2square(box=pseudo_box, w=w, h=h)
    expanded_point = (expanded_box[0], expanded_box[1])
    return expanded_point


def box_xyxy_desquare(box, *, w, h):
    if w == h:
        return box
    if w > h:
        x1, y1, x2, y2 = box
        y1 -= (w - h) // 2
        y2 -= (w - h) // 2
        box = x1, y1, x2, y2
        return box
    assert w < h
    x1, y1, x2, y2 = box
    x1 -= (h - w) // 2
    x2 -= (h - w) // 2
    box = x1, y1, x2, y2
    return box


def renorm_bbox_desquare(text, w, h):
    pattern = r"\[\s*(0\.\d+)\s*,\s*(0\.\d+)\s*,\s*(0\.\d+)\s*,\s*(0\.\d+)\s*\]"

    def replace(match):
        x1, y1, x2, y2 = match.groups()
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

        abs_bbox = de_norm_box_xyxy([x1, y1, x2, y2], w=max(w, h), h=max(w, h))
        abs_bbox_desquare = box_xyxy_desquare(abs_bbox, w=w, h=h)
        new_x1, new_y1, new_x2, new_y2 = norm_box_xyxy(abs_bbox_desquare, w=w, h=h)

        return f"[{new_x1:.2f}, {new_y1:.2f}, {new_x2:.2f}, {new_y2:.2f}]"

    return re.sub(pattern, replace, text)


def renorm_bbox(text, old_w, old_h, new_w, new_h):
    pattern = r"\[\s*(0\.\d+)\s*,\s*(0\.\d+)\s*,\s*(0\.\d+)\s*,\s*(0\.\d+)\s*\]"

    def replace(match):
        x1, y1, x2, y2 = match.groups()
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

        abs_bbox = de_norm_box_xyxy([x1, y1, x2, y2], w=old_w, h=old_h)
        new_x1, new_y1, new_x2, new_y2 = norm_box_xyxy(abs_bbox, w=new_w, h=new_h)

        return f"[{new_x1:.2f}, {new_y1:.2f}, {new_x2:.2f}, {new_y2:.2f}]"

    return re.sub(pattern, replace, text)


def process_box_format(sentence, boxes, boxes_seq):
    # print(sentence, boxes, boxes_seq_str)

    if len(boxes_seq) == 0:
        return sentence

    pos = 0
    for box_group in boxes_seq:
        start = sentence.find("<box>", pos)

        if start == -1:
            break

        replacement = ";".join(map(lambda idx: ",".join(map(str, boxes[idx])), box_group))
        sentence = sentence[:start] + "[" + replacement + "] " + sentence[start + 6:]

        pos = start + len(replacement) + 2

    return sentence
