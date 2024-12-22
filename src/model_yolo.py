
import torch
import torchvision
import torch.nn as nn

BACKBONE_MODELS = {
   'resnet18': {
      'model': torchvision.models.resnet18,
      'weights': torchvision.models.ResNet18_Weights.DEFAULT
   },
   'resnet34': {
      'model': torchvision.models.resnet34,
      'weights': torchvision.models.ResNet34_Weights.DEFAULT
   },
   'resnet50': {
      'model': torchvision.models.resnet50,
      'weights': torchvision.models.ResNet50_Weights.DEFAULT
   },
}

class YOLOV1(nn.Module):
    def __init__(self, model_config):
        super(YOLOV1, self).__init__()
        self.model_config = model_config

        self.yolo_conv_channels = self.model_config['yolo_conv_channels']
        self.leaky_relu_slope = model_config['leaky_relu_slope']
        self.use_batch_norm = model_config['use_batch_norm']
        self.conv_spatial_size = model_config['conv_spatial_size']
        self.fc_dim = model_config['fc_dim']
        self.fc_dropout = model_config['fc_dropout']
        self.use_conv = model_config['use_conv']
        self.use_sigmoid = model_config['use_sigmoid']

        self.S = model_config['S']
        self.B = model_config['B']
        self.C = model_config['C']

        self.normalization = Normalization()

        backbone_model_config = BACKBONE_MODELS[self.model_config['backbone']]
        self.backbone = backbone_model_config['model'](weights=backbone_model_config['weights'])
        self.backbone_channels = self.backbone.fc.in_features

        self.features = nn.Sequential(*list(self.backbone.children())[:-2])

        self.yolo_conv_features = nn.Sequential(
            ConvBlock(self.backbone_channels, self.yolo_conv_channels, kernel_size=3, stride=1, padding=1, negative_slope=self.leaky_relu_slope, use_batch_norm=self.use_batch_norm),
            ConvBlock(self.yolo_conv_channels, self.yolo_conv_channels, kernel_size=3, stride=1, padding=1, negative_slope=self.leaky_relu_slope, use_batch_norm=self.use_batch_norm),
            ConvBlock(self.yolo_conv_channels, self.yolo_conv_channels, kernel_size=3, stride=1, padding=1, negative_slope=self.leaky_relu_slope, use_batch_norm=self.use_batch_norm),
            ConvBlock(self.yolo_conv_channels, self.yolo_conv_channels, kernel_size=3, stride=1, padding=1, negative_slope=self.leaky_relu_slope, use_batch_norm=self.use_batch_norm),
        )

        if self.use_conv:
          self.yolo_detection_head = nn.Sequential(
              nn.Conv2d(self.yolo_conv_channels, 5 * self.B + self.C, kernel_size=1),
          )
        else:
          self.yolo_detection_head = nn.Sequential(
              nn.Flatten(),
              nn.Linear(self.yolo_conv_channels * self.conv_spatial_size * self.conv_spatial_size, self.fc_dim),
              nn.LeakyReLU(self.leaky_relu_slope, inplace=True),
              nn.Dropout(self.fc_dropout),
              nn.Linear(self.fc_dim, self.S * self.S * (5 * self.B + self.C)),
          )



    def forward(self, x):
        out = self.normalization(x)
        out = self.features(out)
        out = self.yolo_conv_features(out)
        out = self.yolo_detection_head(out)
        if self.use_conv:
            out = out.permute(0, 2, 3, 1)
        else:
            out = out.view(out.shape[0], self.S, self.S, (5 * self.B + self.C))

        if self.use_sigmoid:
            # out[..., :5 * self.B] = torch.nn.functional.sigmoid(out[..., :5 * self.B])
            out = torch.nn.functional.sigmoid(out)

        return out


class ConvBlock(nn.Module):
    """
    Convolutional block for YOLOv1
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, negative_slope=0.1, use_batch_norm=True):
        super(ConvBlock, self).__init__()
        self.use_batch_norm = use_batch_norm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        if self.use_batch_norm:
          self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_batch_norm:
          x = self.bn(x)
        x = self.act(x)
        return x



# Create a module to normalize input image so we can easily put it in a
# ``nn.Sequential``
class Normalization(nn.Module):
  def __init__(self, mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])):
    super(Normalization, self).__init__()
    # .view the mean and std to make them [C x 1 x 1] so that they can
    # directly work with image Tensor of shape [B x C x H x W].
    # B is batch size. C is number of channels. H is height and W is width.
    self.mean = mean.view(-1, 1, 1)
    self.std = std.view(-1, 1, 1)
    
    
  def forward(self, img):
    if self.mean.device != img.device:
        self.mean = self.mean.to(img.device)
        self.std = self.std.to(img.device)
    # normalize ``img``
    return (img - self.mean) / self.std