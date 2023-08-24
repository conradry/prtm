import torch

from proteome.models.design.eigenfold.resi_score_model import (
    ResiLevelTensorProductScoreModel,
)


class ScoreModel(torch.nn.Module):
    def __init__(self, args):
        super(ScoreModel, self).__init__()
        self.enn = ResiLevelTensorProductScoreModel(args)

    def forward(self, data):
        return self.enn(data)


def get_model(args):
    return ScoreModel(args)
