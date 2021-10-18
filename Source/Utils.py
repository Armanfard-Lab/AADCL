def MelSpec(y,sr):
  S = torchaudio.transforms.MelSpectrogram(sample_rate= sr, n_fft=2048, hop_length=512, f_max=sr/2 ,norm='slaney',mel_scale='slaney')(y)
  S_dB = torchaudio.transforms.AmplitudeToDB()(S)
  return S_dB
