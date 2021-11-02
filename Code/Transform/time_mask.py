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

def time_mask(S_dB,R=10):
  L_Tmask = int(S_dB.size()[1])/R
  S_Tmask = torchaudio.transforms.TimeMasking(time_mask_param = L_Tmask)(S_dB)
  return S_Tmask
