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

def test(Test,f,Train):
  for X,Y in Train:
    X,Y = apply_transform(X,Y)
    break
  v = model(X)
  v_mean = v.mean(axis=0)
  v_cov = np.cov(np.transpose(v.detach().numpy()))
  v_cov = np.linalg.inv(v_cov)
  v_cov.shape
  y_pred = []
  y_true = []
  H = []
  f.eval()
  for train_features, train_labels in Test:
      X,Y = apply_transform(train_features,train_labels,state=1)
      i=0
      for x in X:
        x = x.reshape(1,1,128,313)
        h = f(x)
        H.append(h)
        y_pred.append(anomaly_score(h.detach().numpy(),v_mean.detach().numpy(),v_cov))
        y_true.append(train_labels.detach().numpy()[i])
        i+=1
