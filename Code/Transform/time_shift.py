import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import librosa
import librosa.display
from MelSpec import MelSpec
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def time_shift(y,shift_max = 5, sr=16000):
  shift_rate = np.random.uniform(0,shift_max)
  shift_rate = int(shift_rate*sr)
  y_shift = np.roll(y.detach().numpy(), shift_rate)
  y_shift = torch.from_numpy(y_shift)
  S = MelSpec(y_shift,sr)
  return S
