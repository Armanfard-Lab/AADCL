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

def fade(y):
  
  shape_var = np.random.randint(0,5)
  if shape_var == 0:
    shape = 'exponential'
  elif shape_var == 1:
    shape = 'logarithmic'
  elif shape_var == 2:
    shape = 'half_sine'
  elif shape_var == 3:
    shape = 'quarter_sine'
  else:
    shape = 'linear'


  L_in = np.random.randint(0,y.shape[0]/2)
  L_out = np.random.randint(0,y.shape[0]/2)

  y_fade = torchaudio.transforms.Fade(fade_in_len=L_in,fade_out_len=L_out,fade_shape=shape)(y)
  S = MelSpec(y_fade,sr)
  return S
