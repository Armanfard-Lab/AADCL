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

def pitch_shift(y,n_steps_min=-10, n_steps_max=10, sr=16000):
  n_steps = np.random.uniform(n_steps_min,n_steps_max)
  y_pitch = librosa.effects.pitch_shift(y.detach().numpy(), sr, n_steps)
  y_pitch = torch.from_numpy(y_pitch)
  S = MelSpec(y_pitch,sr)
  return S
