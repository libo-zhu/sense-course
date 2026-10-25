"""Thin adapters around the MIT-licensed author's actual model classes."""
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from torch import nn

UPSTREAM = Path(__file__).resolve().parent / "upstream"
sys.path.insert(0, str(UPSTREAM))
from models.resnet_ensemble import HSI_Lidar_Couple_Cross_DAD, HSI_Lidar_Couple_Cross_TRI_DAD
from models.single_modality_model import Single_Modality_DAD
from loss.kd.jda import JDA


def create_model(channels: list[int], classes: int):
    args = SimpleNamespace(class_num=classes)
    if len(channels) == 1:
        return Single_Modality_DAD(channels[0], args, pretrained=False)
    if len(channels) == 2:
        return HSI_Lidar_Couple_Cross_DAD(args, *channels)
    if len(channels) == 3:
        return HSI_Lidar_Couple_Cross_TRI_DAD(args, *channels)
    raise ValueError("Only one, two or three modalities are supported")


class SharedSpecificFusion(nn.Module):
    """Paper Eq. (10): freeze features/BatchNorm, train FC heads and scalar eta."""
    def __init__(self, hallucination, regular):
        super().__init__()
        self.hallucination = hallucination
        self.regular = regular
        for model in [hallucination, regular]:
            model.requires_grad_(False)
            model.fc.requires_grad_(True)
        self.eta_logit = nn.Parameter(torch.zeros(()))

    def train(self, mode=True):
        super().train(mode)
        # BatchNorm running statistics are part of the frozen feature extractors.
        self.hallucination.eval()
        self.regular.eval()
        return self

    def forward(self, *inputs):
        h_logits, _ = self.hallucination(*inputs)
        n_logits, _ = self.regular(*inputs)
        eta = self.eta_logit.sigmoid()
        return (1 - eta) * h_logits + eta * n_logits
