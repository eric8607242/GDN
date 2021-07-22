from itertools import combinations
from abc import ABC, abstractmethod

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, criterion_config):
        """
        Args: criterion_config (dict): The config for criterion
        """
        super().__init__()
        self.criterion_config = criterion_config

    def forward(self, x, target):
        """
        Arguments:
            x (torch.Tensor): embeddings of shape (batch, embedding_dim)
            target (torch.Tensor): target labels shape (batch,)

        Return:
            average loss
        """
        return losses.mean()
