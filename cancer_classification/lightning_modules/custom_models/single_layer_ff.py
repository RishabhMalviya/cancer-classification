import torch
import torch.nn as nn


class SingleLayerFF(nn.Module):
    def __init__(self, num_classes=9):
        super(SingleLayerFF, self).__init__()
        self.fc = nn.Linear(224 * 224 * 3, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc(x)
        return x
