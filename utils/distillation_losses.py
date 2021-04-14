import torch
from torch import nn
import torch.nn.functional as F
from typing import List


"""
AttentionMatching:
    Sergey Zagoruyko and Nikos Komodakis.
    Paying more at-tention to attention: Improving the performance of convolutional neural networks via attention transfer.
    International Conference on Learning Representations, 2017.
LogitMatch:
    Geoffrey  Hinton,  Oriol  Vinyals,  and  Jeff  Dean
    Distilling  the  knowledge  in  a  neural  network.
    arXiv  preprintarXiv:1503.02531, 2015.
"""


class AttentionMatching(nn.Module):
    def __init__(
            self,
            weighting: float = 1000,
            indicator: List[int] = None) -> None:
        # - indicator: If it is not None, then it must be a vector of the same length as the inputs to the forward pass
        # It is is a binary vector, 1 indicating the activation map is to be
        # used. None means all are used
        super().__init__()
        self.weighting = weighting
        self.ind = indicator

    def compute_attention(self, x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def forward(self,
                act_maps_s: List[torch.Tensor],
                act_maps_t: List[torch.Tensor]) -> torch.Tensor:
        """
        act_maps_s and act_maps_t are lists of response maps.
        """
        loss = 0
        for i in range(len(act_maps_s)):
            if self.ind is not None and self.ind[i] == 0:
                continue
            act_map_s = act_maps_s[i]
            act_map_t = act_maps_t[i]

            att_map_s = self.compute_attention(act_map_s)
            att_map_t = self.compute_attention(act_map_t)
            loss += (att_map_s - att_map_t).pow(2).mean()

        return self.weighting * loss


class LogitMatch(nn.Module):
    def __init__(
            self,
            T: float = 1,
            weight: float = 1000,
            kl_div: bool = True):
        super().__init__()
        if kl_div:
            self.criterion = nn.KLDivLoss(reduction='mean')
        self.T = T
        self.weight = weight

    def forward(self, output_s, output_t):
        out_s_logsoftmax = F.log_softmax(output_s / self.T, dim=1)
        out_t_softmax = F.softmax(output_t / self.T, dim=1)
        return self.weight * self.criterion(out_s_logsoftmax, out_t_softmax)
