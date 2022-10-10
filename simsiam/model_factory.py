from copy import deepcopy

import torch
from torch import nn
from .resnet_cifar import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


class projection_MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim=4096, output_dim=256):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x


class prediction_MLP(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=4096, output_dim=256):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x


class SimSiam(nn.Module):
    def __init__(self, args):
        super(SimSiam, self).__init__()
        self.backbone = SimSiam.get_backbone(args.arch)
        out_dim = self.backbone.fc.weight.shape[1]
        self.backbone.fc = nn.Identity()

        self.online_projector = projection_MLP(out_dim)
        self.online_encoder = nn.Sequential(
            self.backbone,
            self.online_projector,
        )
        self.online_predictor = prediction_MLP()

        self.target_encoder = deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)

    @staticmethod
    def get_backbone(backbone_name):
        return {'resnet18': ResNet18(),
                'resnet34': ResNet34(),
                'resnet50': ResNet50(),
                'resnet101': ResNet101(),
                'resnet152': ResNet152()}[backbone_name]

    @torch.no_grad()
    def _update_target_network(self, mm):
        """Momentum update of target network"""
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, im_aug1, im_aug2, mm):

        p1 = self.online_predictor(self.online_encoder(im_aug1))
        p2 = self.online_predictor(self.online_encoder(im_aug2))

        with torch.no_grad():
            self._update_target_network(mm)
            z1 = self.target_encoder(im_aug1)
            z2 = self.target_encoder(im_aug2)

        return {'z1': z1, 'z2': z2, 'p1': p1, 'p2': p2}
