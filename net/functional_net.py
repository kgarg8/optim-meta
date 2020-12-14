from torch import nn
import numpy as np
import torch.nn.functional as F
import torch
from typing import Dict

def conv_block(in_channels: int, out_channels: int) -> nn.Module:

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

def functional_conv_block(x: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor, bn_weights, bn_biases) -> torch.Tensor:

    x = F.conv2d(x, weights, biases, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None,
                     weight=bn_weights, bias=bn_biases, training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x


class functional_net(nn.Module):
    def __init__(self, args):
        super().__init__()
        # replace 64 by hidden_channels
        self.conv1 = conv_block(args.in_channels, args.hidden_channels)
        self.conv2 = conv_block(args.hidden_channels, args.hidden_channels)
        self.conv3 = conv_block(args.hidden_channels, args.hidden_channels)
        self.conv4 = conv_block(args.hidden_channels, args.hidden_channels)
        self.logits = nn.Linear(args.hidden_channels * 5 * 5, args.num_way)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)

        return self.logits(x)

    def functional_forward(self, x, weights):

        for block in [1, 2, 3, 4]:
            x = functional_conv_block(x, weights[f'conv{block}.0.weight'], weights[f'conv{block}.0.bias'],
                                      weights.get(f'conv{block}.1.weight'), weights.get(f'conv{block}.1.bias'))

        x = x.view(x.size(0), -1)

        x = F.linear(x, weights['logits.weight'], weights['logits.bias'])

        return x
