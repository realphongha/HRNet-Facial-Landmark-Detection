import torch
import torch.nn as nn
from torch.nn import functional
from .hrnet import HighResolutionNet


class Net2(nn.Module):
    def __init__(self, backbone, n_points, n_classes=3):
        super().__init__()
        n = n_points
        self.backbone = backbone
        self.map_to_pts = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64, n * 2),
            nn.ReLU(inplace=True),
            nn.Linear(n * 2, n * 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n*2, n*8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(n*8, n*2),
            nn.ReLU(inplace=True),
            nn.Linear(n*2, n_classes),
        )

    def forward(self, x, meta):
        x = self.backbone(x)
        # score_map = x.data
        x = torch.sum(x, dim=1)
        x = self.map_to_pts(x)
        x = self.classifier(x)
        return x


def hrnet_pose(config, **kwargs):
    hrnet = HighResolutionNet(config, **kwargs)
    if config["MODEL"]["FREEZE_BACKBONE"]:
        for name, p in hrnet.named_parameters():
            print("Freezing %s..." % name)
            p.requires_grad = False
        print("Froze HRNet's weights")
    pretrained = config.MODEL.PRETRAINED if config.MODEL.INIT_WEIGHTS else ''
    hrnet.init_weights(pretrained=pretrained)
    model = Net2(backbone=hrnet, n_points=config.MODEL.POSE_POINTS)
    if config["MODEL"]["FREEZE_CLF"]:
        for name, p in model.named_parameters():
            if "map_to_pts" in name or "classifier" in name:
                print("Freezing %s..." % name)
                p.requires_grad = False
        print("Froze classifier weights")
    return model




