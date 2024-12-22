import os

import numpy as np
import torch

from torch.utils.data.dataset import Dataset
from torchvision import datasets


class VOCDataset(Dataset):
  def __init__(self, image_set='train', year='2007', root='./VOC', transform=None,
               S=7, B=2, C=20, return_all=False):

    self.dataset = datasets.VOCDetection(
        root=root,
        year=year,
        image_set=image_set,
        download= not os.path.exists(root),
    )

    self.return_all = return_all
    self.transform = transform
    self.S = S
    self.B = B
    self.C = C

    classes = [
        'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
        'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
        'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
    ]
    classes = sorted(classes)
    self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
    self.idx2label = {idx: classes[idx] for idx in range(len(classes))}

  def extract_bboxes(self, annotation):
    objects = annotation['annotation']['object']

    bboxes = []
    labels = []

    for obj in objects:
        bndbox = obj['bndbox']
        bbox_coords = [int(c) for c in bndbox.values()]
        bboxes.append(bbox_coords)
        labels.append(self.label2idx[obj['name']])

    return bboxes, labels

  def construct_yolo_target(self, image_size, bboxes_tensor, labels_tensor):
    # Build Target for Yolo
    target_dim = 5 * self.B + self.C
    h, w = image_size
    yolo_targets = torch.zeros(self.S, self.S, target_dim)

    # Height and width of grid cells is H // S
    cell_pixels_h = h // self.S
    cell_pixels_w = w // self.S

    # print(cell_pixels_h, cell_pixels_w)

    if len(bboxes_tensor) > 0:
        # Convert x1y1x2y2 to xywh format
        box_widths = bboxes_tensor[:, 2] - bboxes_tensor[:, 0]
        box_heights = bboxes_tensor[:, 3] - bboxes_tensor[:, 1]
        box_center_x = bboxes_tensor[:, 0] + 0.5 * box_widths
        box_center_y = bboxes_tensor[:, 1] + 0.5 * box_heights

        # print(box_center_x, box_center_y)

        # Get cell i,j from xc, yc
        box_j = torch.floor(box_center_x / cell_pixels_w).long()
        box_i = torch.floor(box_center_y / cell_pixels_h).long()

        # print(box_i, box_j)

        # xc offset from cell topleft
        box_xc_cell_offset = (box_center_x - box_j*cell_pixels_w) / cell_pixels_w
        box_yc_cell_offset = (box_center_y - box_i*cell_pixels_h) / cell_pixels_h

        # print(box_xc_cell_offset, box_yc_cell_offset)

        # w, h targets normalized to 0-1
        box_w_label = box_widths / w
        box_h_label = box_heights / h

        # print(box_w_label, box_h_label)

        # Update the target array for all bboxes
        for idx in range(len(bboxes_tensor)):
            # # Make target of the exact same shape as prediction
            for k in range(self.B):
                s = 5 * k
                # target_ij = [xc_offset,yc_offset,sqrt(w),sqrt(h), conf, cls_label]
                yolo_targets[box_i[idx], box_j[idx], s] = box_xc_cell_offset[idx]
                yolo_targets[box_i[idx], box_j[idx], s+1] = box_yc_cell_offset[idx]
                yolo_targets[box_i[idx], box_j[idx], s+2] = box_w_label[idx]
                yolo_targets[box_i[idx], box_j[idx], s+3] = box_h_label[idx]
                yolo_targets[box_i[idx], box_j[idx], s+4] = 1.0

            label = int(labels_tensor[idx])
            cls_target = torch.zeros((self.C,))
            cls_target[label] = 1.
            yolo_targets[box_i[idx], box_j[idx], 5 * self.B:] = cls_target


    return yolo_targets

  def __getitem__(self, index):
    sample = self.dataset[index]
    image, annotation = sample

    image = np.array(image)
    bboxes, labels = self.extract_bboxes(annotation)

    if self.transform is not None:
      transformed_sample = self.transform(image=image, bboxes=bboxes, labels=labels)
      image, bboxes, labels = transformed_sample['image'], transformed_sample['bboxes'], transformed_sample['labels']

    
    image_tensor = torch.from_numpy(image / 255.).permute((2, 0, 1)).float()
    bboxes_tensor = torch.as_tensor(bboxes)
    labels_tensor = torch.as_tensor(labels)

    yolo_targets = self.construct_yolo_target(image_tensor.shape[1:], bboxes_tensor, labels_tensor)

    sample = {
        'image_tensor': image_tensor,
        'yolo_targets': yolo_targets
    }

    if self.return_all:
       sample['image'] = image
       sample['bboxes'] = bboxes
       sample['labels'] = labels

    return sample

  def __len__(self):
    return len(self.dataset)

