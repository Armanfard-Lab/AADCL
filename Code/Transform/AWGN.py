import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import librosa
import librosa.display
from MelSpec import MelSpec

def AWGN(y,SNR_min=-6, SNR_max=6, sr=16000):
  SNR = np.random.uniform(SNR_min,SNR_max)
  y = y.detach().numpy()
  RMS = np.sqrt(np.mean(y**2))
  STD_n = np.sqrt(RMS**2/(10**(SNR/10)))
  noise = np.random.normal(0,STD_n,y.shape[0])
  y_noise = y + noise
  y_noise = torch.from_numpy(y_noise)
  S = MelSpec(y_noise.float(),sr)
  return S
