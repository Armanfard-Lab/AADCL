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

def MelSpec(y,sr):
  S = torchaudio.transforms.MelSpectrogram(sample_rate= sr, n_fft=2048, hop_length=512, f_max=sr/2 ,norm='slaney',mel_scale='slaney')(y)
  S_dB = torchaudio.transforms.AmplitudeToDB()(S)
  return S_dB