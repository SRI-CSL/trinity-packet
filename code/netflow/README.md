## Context PCAP

cnn.py: Discriminative CNN model for payload classification.


dataset.py: 

    ContextPCAPDataset: class for PCAP dataset (CIC, UNSW, etc.). Inherits from data_structures/dataset.py:NetflowDataset.

    ContextPCAPTorchDataset: class for PyTorch DataLoader. 


experiment.py: 

    ContextPCAPExperiment: class for experiment (Baseline, OOD, etc.). Inherits from data_structures/dataset.py:NetflowExperiment.


fnn.py: Descriminative FNN model for payload classification.


preproces.py: Functions to transform PCAP + flow files into usable datasets. 


transformer.py: Descriminative Transformer model for payload classification.


## Data Structures

dataset.py:

    NetflowDataset: parent class for othery types of datasets (ContextPCAP, SequencePCAP, etc.).


experiment.py:

    NetflowExperiment: parent class that contains functions for training, inference, OOD experiments.


network_model.py:

    NetworkModel: wrapper class for training/inference/feature extraction.


util.py: Miscellaneous parsing functions.

## Detectors

iaf.py:

    IAFDataset: A normalizing flows implementation that allows for model distributions.
