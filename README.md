# CE-RPCANet

This repository provides the official PyTorch implementation of CE-RPCANet
for infrared small target detection, including training, testing, and
evaluation code.

## Models

- `ce_rpcanet`: the complete six-stage CE-RPCANet with AOCM.
- `ce_rpcanet_without_aocm`: the same network without AOCM.

The implementation uses the terminology from the paper: Global–Local
Background Update, Global Context Branch, Lightweight Context Encoder,
Residual-Response Sparse Prior Generator, Sparse Support Refinement,
Kernelized Sparse Attention, Prior-Guided Sparse Object Update, and Adapted
Observation-Consistency Module.

## Installation

```bash
pip install -r requirements.txt
```

## Dataset layout

Place datasets under `datasets/` as described in `datasets/README.md`.

## Training

```bash
python train.py --net-name ce_rpcanet --dataset nudt --gpu 0
python train.py --net-name ce_rpcanet --dataset irstd1k --gpu 0
python train.py --net-name ce_rpcanet --dataset sirst --gpu 0
```

To train the model without AOCM, use
`--net-name ce_rpcanet_without_aocm`.

## Testing and evaluation

Set the checkpoint and dataset paths in `test.py`, then run:

```bash
python test.py
python evaluate.py
```
