import pytorch_lightning as pl
import torchmetrics
from torchmetrics import JaccardIndex,Dice
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.distributions import normal
from torch.optim import SGD
import torch.nn.functional as F

def compute_metrics(pl_module: pl.LightningModule):
    pass
