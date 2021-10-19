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

def pitch_shift(y,n_steps_min=-10, n_steps_max=10, sr=16000):
  n_steps = np.random.uniform(n_steps_min,n_steps_max)
  y_pitch = librosa.effects.pitch_shift(y.detach().numpy(), sr, n_steps)
  y_pitch = torch.from_numpy(y_pitch)
  S = MelSpec(y_pitch,sr)
  return S
