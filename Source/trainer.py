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


def trainer(Train,f=resnet18().to(device),g=Projection().to(device),CLS=LinCLS().to(device),NTXent=losses.NTXentLoss(), CELoss=torch.nn.CrossEntropyLoss(),num_epochs=200,verbosity=0,pretrain=True):
  if pre_train == False:
    optimizer = torch.optim.Adam([
                    {'params': f.parameters()},
                    {'params': g.parameters()},
                    {'params': CLS.parameters(),}
                ], lr=1e-2)


    for epoch in range(num_epochs):
      epoch_loss = 0
      cls_loss = 0
      num_batch = 0
      for train_features, train_labels in Train:
        X,Y = apply_transform(train_features,train_labels)
        optimizer.zero_grad()
        X = X.to(device)
        Y = Y.to(device)
        Y_NTXent = torch.arange(X.shape[0])
        Y_NTXent[int(X.shape[0]/2):] = Y_NTXent[0:int(X.shape[0]/2)]
        h = f(X)
        z = g(h)
        cls_pred = CLS(h)
        loss = NTXent(z,Y_NTXent) + 0.1*CELoss(cls_pred, Y.long())

        loss.backward()
        optimizer.step()
        epoch_loss += loss
        #epoch_loss += NTXent(z,Y_NTXent)
        #cls_loss += CELoss(cls_pred, Y.long())
        num_batch += 1
      if verbosity>0:
        #print("epoch : {}/{}, CLS loss = {:.6f}".format(epoch + 1, num_epochs, cls_loss/num_batch))
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, epoch_loss/num_batch))
  else:
    PATH = "state_dict_model.pt"
    f.load_state_dict(torch.load(PATH))
