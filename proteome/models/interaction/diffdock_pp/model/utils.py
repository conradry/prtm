import torch.nn as nn


def _init(model):
    for name, param in model.named_parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
