from torch import nn
from torch.nn import LeakyReLU, CELU, SELU
from torch.nn import functional as F

LeakyReLU
CELU
SELU


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)
