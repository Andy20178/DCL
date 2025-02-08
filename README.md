# 🚀 Source Code for Robust Disentangled Counterfactual Learning for Physical Commonsense Reasoning (NeurIPS 2023) 🧠

Welcome to the official repository for our NeurIPS 2023 paper! This README will guide you through setting up and running our experiments. Let's dive in! 🚀

## 📥 Downloading Model Weights

First, you need to download the pretrained CLIP and AudioCLIP weights.

### CLIP
```bash
wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

### AudioCLIP
Download `AudioCLIP-Partial-Training.py` and `bpe_simple_vocab_16e6.txt.gz` from [AudioCLIP Releases](https://github.com/AndreyGuzhov/AudioCLIP/releases).

Once downloaded, place the models into the `assets` folder.

## 🛠️ Requirements

We use the following setup:
- **Python**: 3.8.10
- **PyTorch**: 1.11.0
- **CUDA**: 11.3

## 🏋️‍♂️ Training

There are two models that can be trained:

### 1. PACS Dataset
```bash
conda activate PACS
python3 train_1.py
```

### 2. Material Classification Dataset
```bash
python3 train_classify.py
```

## 🔮 Prediction

Once a model has been trained, you can generate predicted outputs on the test set:

```bash
python3 predict.py -model_path PATH_TO_MODEL_WEIGHTS -split test
```

## 🙏 Acknowledgements

This code was adapted from:
- [AudioCLIP](https://github.com/AndreyGuzhov/AudioCLIP)
- [PACS](https://github.com/samuelyu2002/PACS)

We would like to thank Andrey Guzhov, Samuel Yu, and all other contributors of these repositories for their valuable work. 🙌

---

Feel free to reach out if you have any questions or need further assistance! 🚀