from torch import nn


class SimSiamLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.criterion = nn.CosineSimilarity(dim=1)

    def forward(self, z1, z2, p1, p2):

        loss1 = - self.criterion(p1, z2).mean()
        loss2 = - self.criterion(p2, z1).mean()

        return 0.5 * loss1 + 0.5 * loss2
