# AADCL: Self-Supervised Acoustic Anomaly Detection Via Contrastive Learning
PyTorch implementation of the paper "Semi-Supervised Acoustic Anomaly Detection via Contrastive Learning"

<center><img src="https://github.com/Armanfard-Lab/AADCL/blob/main/Figs/Overview.jpg" alt="Overview" width="800" align="center"></center>

## Citation

You can find our paper on [IEEE Xplore](https://ieeexplore.ieee.org/document/9746207).

Please cite our paper if you use the results or codes of our work.

```
H. Hojjati and N. Armanfard, "Self-Supervised Acoustic Anomaly Detection Via Contrastive Learning," ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Singapore, Singapore, 2022, pp. 3253-3257, doi: 10.1109/ICASSP43922.2022.9746207.
```

## Abstract

>We propose an acoustic anomaly detection algorithm based on the framework of contrastive learning. Contrastive learning is a recently proposed self-supervised approach that has shown promising results in image classification and speech recognition. However, its application in anomaly detection is underexplored. Earlier studies have demonstrated that it can achieve state-of-the-art performance in image anomaly detection, but its capability in anomalous sound detection is yet to be investigated. For the first time, we propose a contrastive learning-based framework that is suitable for acoustic anomaly detection. Since most existing contrastive learning approaches are targeted toward images, the effect of other data transformations on the performance of the algorithm is unknown. Our framework learns a representation from unlabeled data by applying audio-specific data augmentations. We show that in the resulting latent space, normal and abnormal points are distinguishable. Experiments conducted on the MIMII dataset confirm that our approach can outperform competing methods in detecting anomalies.



