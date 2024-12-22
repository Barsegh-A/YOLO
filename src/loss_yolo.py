import torch
import torch.nn as nn

from .utils import get_iou

class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        """
        Args:
            S: Grid size (e.g., 7 for 7x7 grid).
            B: Number of bounding boxes per grid cell.
            C: Number of classes.
            lambda_coord: Weight for bounding box regression loss.
            lambda_noobj: Weight for no-object confidence loss.
        """
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def convert_box_coords(self, boxes):
        """
        Convert box coordinates from (x_c, y_c, w, h) to (x1, y1, x2, y2),
         where x_c and y_c are the bbox center coordinates relative to the cell and w, h are width and height realtive to the whole image,
         and x1, y1, x2, y2 are the bbox coordinates relative to the image.
        """

        # Shifts for all grid cell locations.
        # Will use these for converting x_center_offset/y_center_offset
        # values to x1/y1/x2/y2(normalized 0-1)
        # S cells = 1 => each cell adds 1/S pixels of shift
        shifts_x = torch.arange(0, self.S,
                        dtype=torch.int32,
                        device=boxes.device) * 1 / float(self.S)
        shifts_y = torch.arange(0, self.S,
                                dtype=torch.int32,
                                device=boxes.device) * 1 / float(self.S)

        # Create a grid using these shifts
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        # shifts -> (1, S, S, B)
        shifts_x = shifts_x.reshape((1, self.S, self.S, 1)).repeat(1, 1, 1, self.B)
        shifts_y = shifts_y.reshape((1, self.S, self.S, 1)).repeat(1, 1, 1, self.B)

        # xc_offset yc_offset w h -> x1 y1 x2 y2 (normalized 0-1)
        # x_center = (xc_offset / S + shift_x)
        # x1 = x_center - 0.5 * w
        # x2 = x_center + 0.5 * w
        boxes_x1 = ((boxes[..., 0]/self.S + shifts_x)
                         - 0.5*boxes[..., 2])
        boxes_x1 = boxes_x1[..., None]

        boxes_y1 = ((boxes[..., 1]/self.S + shifts_y)
                         - 0.5*boxes[..., 3])
        boxes_y1 = boxes_y1[..., None]

        boxes_x2 = ((boxes[..., 0]/self.S + shifts_x)
                         + 0.5*boxes[..., 2])
        boxes_x2 = boxes_x2[..., None]

        boxes_y2 = ((boxes[..., 1]/self.S + shifts_y)
                         + 0.5*boxes[..., 3])
        boxes_y2 = boxes_y2[..., None]

        boxes_x1y1x2y2 = torch.cat([
            boxes_x1,
            boxes_y1,
            boxes_x2,
            boxes_y2], dim=-1)

        return boxes_x1y1x2y2


    def forward(self, predictions, targets):
        """
        Compute the YOLO loss.
        Args:
            predictions: Tensor of shape (batch_size, S, S, B * 5 + C).
            targets: Tensor of shape (batch_size, S, S, B * 5 + C).
        Returns:
            Total loss.
        """

        # print(predictions.shape, targets.shape)

        # # Split predictions
        pred_boxes = predictions[..., :self.B * 5].view(-1, self.S, self.S, self.B, 5)
        pred_classes = predictions[..., self.B * 5:]  # Class probabilities

        # print(pred_boxes.shape, pred_classes.shape)

        # Split targets
        target_boxes = targets[..., :self.B * 5].view(-1, self.S, self.S, self.B, 5)
        target_classes = targets[..., self.B * 5:]  # Class probabilities

        # print(target_boxes.shape, target_classes.shape)

        # Extract box components
        pred_conf = pred_boxes[..., -1]  # Confidence score
        pred_coords = pred_boxes[..., :-1]  # x, y, w, h

        # print(pred_conf.shape, pred_coords.shape)

        target_conf = target_boxes[..., -1]  # Confidence score
        target_coords = target_boxes[..., :-1]  # x, y, w, h

        # print(target_conf.shape, target_coords.shape)

        # Object mask (1 if object exists, 0 otherwise)
        is_object = target_conf > 0 # bs x bs x s x b
        is_object = is_object[..., 0]
        is_object = is_object[..., None]

        # print(is_object.shape)

        pred_coords_x1y1x2y2 = self.convert_box_coords(pred_coords)
        target_coords_x1y1x2y2 = self.convert_box_coords(target_coords)

        # iou -> (Batch_size, S, S, B)
        iou = get_iou(pred_coords_x1y1x2y2, target_coords_x1y1x2y2)

        # max_iou_val/max_iou_idx -> (Batch_size, S, S, 1)
        max_iou_val, max_iou_idx = iou.max(dim=-1, keepdim=True)

        #########################
        # Indicator Definitions #
        #########################
        # before max_iou_idx -> (Batch_size, S, S, 1) Eg [[0], [1], [0], [0]]
        # after repeating max_iou_idx -> (Batch_size, S, S, B)
        # Eg. [[0, 0], [1, 1], [0, 0], [0, 0]] assuming B = 2
        max_iou_idx = max_iou_idx.repeat(1, 1, 1, self.B)
        # bb_idxs -> (Batch_size, S, S, B)
        #  Eg. [[0, 1], [0, 1], [0, 1], [0, 1]] assuming B = 2
        bb_idxs = (torch.arange(self.B).reshape(1, 1, 1, self.B).expand_as(max_iou_idx)
                   .to(pred_coords.device))
        # is_max_iou_box -> (Batch_size, S, S, B)
        # Eg. [[True, False], [False, True], [True, False], [True, False]]
        # only the index which is max iou boxes index will be 1 rest all 0
        is_max_iou_box = (max_iou_idx == bb_idxs).long()

        is_max_box_obj = is_max_iou_box*is_object


        #####################
        # Localization Loss #
        #####################
        x_mse = (pred_boxes[..., 0] - target_boxes[..., 0]) ** 2
        x_mse = (is_max_box_obj * x_mse).sum()

        y_mse = (pred_boxes[..., 1] - target_boxes[..., 1]) ** 2
        y_mse = (is_max_box_obj * y_mse).sum()

        w_sqrt_mse = (pred_boxes[..., 2]**0.5 - target_boxes[..., 2]**0.5) ** 2
        w_sqrt_mse = (is_max_box_obj * w_sqrt_mse).sum()

        h_sqrt_mse = (pred_boxes[..., 3]**0.5 - target_boxes[..., 3]**0.5) ** 2
        h_sqrt_mse = (is_max_box_obj * h_sqrt_mse).sum()

        localization_loss = self.lambda_coord*(x_mse + y_mse + w_sqrt_mse + h_sqrt_mse)

        # print('localization', localization_loss, x_mse, y_mse, w_sqrt_mse, h_sqrt_mse)

        ######################################################
        # Objectness Loss (For responsible predictor boxes ) #
        ######################################################
        # indicator is now object_cells * is_best_box
        obj_mse = (iou - pred_conf) ** 2
        # Only keep losses from boxes of cells with object assigned
        # and that box which is the responsible predictor
        obj_mse = (is_max_box_obj * obj_mse).sum()

        # print('obj_mse', obj_mse)

        ######################################################

        #################################################
        # Objectness Loss
        # For boxes of cells assigned with object that
        # aren't responsible predictor boxes
        # and for boxes of cell not assigned with object
        #################################################
        no_object_indicator = 1 - is_max_box_obj
        no_obj_mse = (torch.zeros_like(pred_conf) - pred_conf) ** 2
        no_obj_mse = (no_object_indicator * no_obj_mse).sum()
        no_obj_mse = self.lambda_noobj * no_obj_mse

        # print('no_obj_mse', no_obj_mse)


        # 3. Classification Loss

        # print(pred_classes, target_classes)

        class_loss = torch.sum(
            is_object * (pred_classes - target_classes) ** 2
        )
        # print('classification', class_loss)

        # Total Loss
        # localization_loss = localization_loss / 10
        # class_loss = class_loss / 10
        # print(localization_loss.item(), obj_mse.item(), no_obj_mse.item(), class_loss.item())

        total_loss = localization_loss + obj_mse + no_obj_mse + class_loss
        total_loss = total_loss / predictions.shape[0]

        return total_loss