# Deep Learning Based Network Intrusion Anomaly Detection

Unsupervised anomaly detection for network-traffic and multivariate time-series data. This repository contains the code from my undergraduate thesis, **"Deep Learning Based Network Intrusion Anomaly Detection Algorithm"**, which proposes **TadAGAN**: an attention-enhanced variant of TadGAN, and compares it against TadGAN and Anomaly Transformer.

> **Status:** research code, cleaned up for sharing. Datasets and trained model weights are **not** included (see [Data](#data)).

## Overview

| Component | Location | Description |
|---|---|---|
| **TadAGAN** (thesis model) | `tadgan/TadAGAN.py` | TadGAN with self-attention in the generator and input dropout in the encoder |
| TadGAN variants | `tadgan/tadGAN.py`, `tadgan/lstm_tadgan.py` | Baseline / alternative variants used for comparison |
| Data loaders | `tadgan/dataload.py` | Loaders for NSL-KDD, TON_IoT, SMD, MSL, SMAP, PSM and SWaT |
| Preprocessing | `tadgan/preprocess/` | Scripts that convert raw TON_IoT / SWaT / SMAP files into the expected format |
| Anomaly Transformer | `anomaly_transformer/` | Anomaly Transformer baseline (modified from the official implementation) |
| Results | `results/` | Experiment logs and result tables |

## TadAGAN architecture

TadAGAN keeps the TadGAN framework (encoder-generator pair trained against two Wasserstein critics) and changes the following:

- **Encoder:** input dropout (p = 0.2) followed by a bidirectional LSTM, flattened and projected to the latent space.
- **Generator:** bidirectional LSTM, temporal upsampling, then a **multi-head self-attention block** (4 heads, residual connection, LayerNorm, dropout 0.2), a second bidirectional LSTM, and a time-distributed linear output with `tanh`.
- **Critics:** Critic-X (1D convolutions) scores real vs. reconstructed windows; Critic-Z (MLP) scores latent codes. Both are trained with a Wasserstein loss and gradient penalty, with 3 critic updates per generator update.
- **Detection:** at test time the critic score and reconstruction error are combined into an anomaly score, and a sliding-window local threshold is applied. Reported metrics: accuracy, precision, recall, F1 and ROC-AUC.

## Installation

```bash
git clone https://github.com/Chenric-Ni/deep-learning-network-intrusion-detection.git
cd deep-learning-network-intrusion-detection
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The code was developed with Python 3.8 and PyTorch.

## Data

No datasets are redistributed here. See [`data/README.md`](data/README.md) for where to download each dataset and the folder layout the loaders expect.

## Usage

Run from the `tadgan/` directory.

**Train TadAGAN on NSL-KDD**

```bash
cd tadgan
python main.py --mode train --model_name tadagan --dataset KDD \
    --data_path ../data/NSL-KDD --model_save_path ../checkpoints \
    --input_dim 41 --win_size 50 --num_epochs 800
```

**Evaluate a trained model**

```bash
python main.py --mode test --model_name tadagan --dataset KDD \
    --data_path ../data/NSL-KDD --model_save_path ../checkpoints
```

Test mode loads `<dataset>_best_model.pth` from `--model_save_path`, which training writes whenever validation loss improves. Set `--input_dim` to match the number of features in your dataset. Supported `--dataset` values: `KDD`, `TON`, `SMD`, `MSL`, `SMAP`, `PSM`, `SWAT`.

**Anomaly Transformer baseline**

```bash
cd anomaly_transformer
python main.py --mode train --dataset SMD --data_path ../data/SMD
```

See `anomaly_transformer/UPSTREAM_README.md` and `anomaly_transformer/scripts/` for the original per-dataset commands.

## Results

<!-- TODO: fill this table from the thesis. Only report numbers that appear in the thesis. -->

| Dataset | Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| NSL-KDD | TadGAN |0.588|0.501|0.541|0.515|
| NSL-KDD | **TadAGAN** |0.602|0.528|0.563|0.592|
| NSL-KDD | Anomaly Transformer |0.715|0.379|0.469|0.560|
| TON_IoT | TadGAN |0.491|0.438|0.489|0.599|
| TON_IoT | **TadAGAN** |0.600|0.548|0.572|0.581|
| TON_IoT | Anomaly Transformer |0.490|0.486|0.493|0.481|
| SMD | TadGAN |0.533|0.655|0.600|0.917|
| SMD | **TadAGAN** |0.534|0.704|0.611|0.915|
| SMD | Anomaly Transformer |0.381|0.718|0.498|0.840|

Full experimental details are in [`docs/thesis_summary.md`](docs/thesis_summary.md).

## Limitations

- Datasets and pretrained weights are not included, so results must be reproduced by retraining.
- Some scripts under `tadgan/preprocess/` were written for one-off dataset conversion and may need path adjustments.
- Hyperparameters are set through command-line defaults; no automated tuning is included.

## Attribution and license

This project builds on two open-source implementations. Their original licenses are preserved in [`third_party_licenses/`](third_party_licenses/).

- **TadGAN:** Geiger et al., *TadGAN: Time Series Anomaly Detection Using Generative Adversarial Networks* ([arXiv:2009.07769](https://arxiv.org/abs/2009.07769)). This code is based on an open-source PyTorch implementation of the paper.
- **Anomaly Transformer:** Xu et al., *Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy* (ICLR 2022), [official implementation](https://github.com/thuml/Anomaly-Transformer).

The attention-based modifications (TadAGAN), data loaders, preprocessing scripts, and experiments are my own work for the thesis.

Modifications in this repository are released under the MIT License. See `LICENSE`.
