import torch
import torch.nn as nn


# Final attention layer before classification network
# as used by "Human Activity Recognition from Wearable Sensor Data Using Self-Attention"
# Saif Mahmud and M. Tanjid Hasan Tonmoy et al.
class GlobalTemporalAttention(nn.Module):
    def __init__(self, model_dim, dropout=0.0):
        super().__init__()

        self.linear_tanh = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout),
            nn.Tanh(),
        )

        self.g_net = nn.Linear(model_dim, 1)

    def forward(self, x, mask=None):
        uit = self.linear_tanh(x)
        ait = self.g_net(uit)
        a = torch.exp(ait)

        if mask is not None:
            a *= mask.unsqueeze(2)
        
        a = a / ( torch.sum(a, dim=1, keepdim=True) + torch.finfo(torch.float32).eps )
        weighted_input = x * a
        result = torch.sum(weighted_input, dim=1)
        return result