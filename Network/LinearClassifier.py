import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader,ConcatDataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as signal
import librosa
import librosa.display
from scipy.io import wavfile
from IPython.display import Audio
from pytorch_metric_learning import losses
import scipy
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinCLS(nn.Module):
    def __init__(self, input_dim=512, output_dim=8):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x
