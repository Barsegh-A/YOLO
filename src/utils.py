import cv2
import torch

def visualize_bbox(img, bbox, class_name, score=None, color=(255, 0, 0), thickness=2):
    r"""
    Draws a single bounding box on image
    """
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max )
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    box_text = class_name + ' {:.2f}'.format(score) if score is not None else class_name
    ((text_width, text_height), _) = cv2.getTextSize(box_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=box_text,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.45,
        color=(255, 255, 255),
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name, scores=None):
    img = image.copy()
    for idx, (bbox, category_id) in enumerate(zip(bboxes, category_ids)):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name, scores[idx] if scores is not None else None)
    return img



def get_iou(boxes1, boxes2):
    r"""
    IOU between two sets of boxes
    """
    # Area of boxes (x2-x1)*(y2-y1)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # Get top left x1,y1 coordinate
    x_left = torch.max(boxes1[..., 0], boxes2[..., 0])
    y_top = torch.max(boxes1[..., 1], boxes2[..., 1])

    # Get bottom right x2,y2 coordinate
    x_right = torch.min(boxes1[..., 2], boxes2[..., 2])
    y_bottom = torch.min(boxes1[..., 3], boxes2[..., 3])

    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)
    union = area1.clamp(min=0) + area2.clamp(min=0) - intersection_area
    iou = intersection_area / (union + 1E-6)
    return iou