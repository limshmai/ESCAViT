## ESCAViT: Symmetry-Aware EEG Classification

This repository contains the implementation of ESCAViT for IIIC pattern classification, including:

AES_Mix.py: Implementation of Adaptive EEG Spectrogram Mixup (AES-Mix)

LIGCL.py: Lead Interrelation-Guided Contrastive Learning (LIGCL) Loss

ESCAViT.py: ESCAViT model architecture


## Overview
Adaptive EEG Spectrogram Mixup (AES-Mix): Custom mixup strategy for handling EEG data ambiguity

Lead Interrelation-Guided Contrastive Learning (LIGCL): Loss function designed for lead symmetry learning

ESCAViT: Multi-stream architecture combining ViViT with lead attention mechanisms


## Dataset
This study used data from the "HMS - Harmful Brain Activity Classification" competition on Kaggle (https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification). The dataset is licensed under CC BY-NC 4.0.


## Training Setting
All models were trained/tested on one NVIDIA GeForce RTX A6000 48GB GPU. Training is facilitated by the AdamW optimizer, with a learning rate of 1e-4, weight decay of 1e-3, and is conducted over 50 epochs with a batch size of 8.
