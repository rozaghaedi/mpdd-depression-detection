## MPDD: Multimodal Personalized Depression Detection

Multimodal Depression Detection for Elderly and Young Adults

## Overview

This project implements a multimodal deep learning system for depression detection using:
- Audio features (MFCC, Wav2Vec, OpenSMILE)
- Visual features (ResNet, DenseNet, OpenFace) 
- Personalized text embeddings
- Advanced attention mechanisms

## Installation

```bash
pip install -r requirements.txt
```



## Quick Start

```python
from src.config import MPDDConfig
from src.utils import run_experiment
import torch

# Setup
config = MPDDConfig()
config.elderly_data_path = "/path/to/MPDD-Elderly"
config.young_data_path = "/path/to/MPDD-Young"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Run experiment
result = run_experiment(
    dataset_name="elderly",
    window_size="5s", 
    task_type="binary",
    audio_feature="wav2vec",
    visual_feature="resnet",
    config=config,
    device=device
)
```

## Project Structure

```
mpdd-depression-detection/
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── models.py
│   ├── training.py
│   └── utils.py
├── example_usage.py
├── requirements.txt
└── README.md
```

## Features

- Multi-modal fusion with attention mechanisms
- Support for elderly and young adult datasets
- Binary, ternary, and quinary classification
- Robust error handling
- Pre-extracted feature support

## Usage

Run the example:
```bash
python example_usage.py
```
