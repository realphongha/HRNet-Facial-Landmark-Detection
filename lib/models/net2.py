import torch
import torch.nn as nn
from .hrnet import HighResolutionNet
from .debug import PrintLayer


class Net2(nn.Module):
    def __init__(self, backbone, n_points, n_classes=3):
        super().__init__()
        self.backbone = backbone
        # self.map_to_pts = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(64 * 64, n_points * 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(n_points * 2, n_points * 2),
        # )
        self.map_to_pts = nn.Sequential(
            nn.Conv2d(in_channels=n_points, out_channels=1, kernel_size=(3, 3), padding=1),
            # PrintLayer("shape", True),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64*64, n_points * 2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_points * 2, n_points * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(n_points * 8, n_points * 2),
            nn.ReLU(inplace=True),
            nn.Linear(n_points * 2, n_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        # x = torch.sum(x, dim=1)
        x = self.map_to_pts(x)
        x = self.classifier(x)
        return x

    def freeze_weights(self, freeze_backbone, freeze_clf):
        if freeze_backbone:
            for name, p in self.named_parameters():
                if "backbone" in name:
                    print("Freezing %s..." % name)
                    p.requires_grad = False
            print("Froze backbone weights")
        elif freeze_clf:
            for name, p in self.named_parameters():
                if "map_to_pts" in name or "classifier" in name:
                    print("Freezing %s..." % name)
                    p.requires_grad = False
            print("Froze classifier weights")



def hrnet_pose(config, **kwargs):
    hrnet = HighResolutionNet(config, **kwargs)
    # if config["MODEL"]["FREEZE_BACKBONE"]:
    #     for name, p in hrnet.named_parameters():
    #         print("Freezing %s..." % name)
    #         p.requires_grad = False
    #     print("Froze HRNet's weights")
    pretrained = config.MODEL.BACKBONE_PRETRAINED if config.MODEL.INIT_WEIGHTS else ''
    hrnet.init_weights(pretrained=pretrained)
    model = Net2(backbone=hrnet, n_points=config.MODEL.NUM_JOINTS)
    # if config["MODEL"]["FREEZE_CLF"]:
    #     for name, p in model.named_parameters():
    #         if "map_to_pts" in name or "classifier" in name:
    #             print("Freezing %s..." % name)
    #             p.requires_grad = False
    #     print("Froze classifier weights")
    return model




