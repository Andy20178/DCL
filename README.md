# 🚀 Source Code for Disentangled Counterfactual Learning for Physical Commonsense Reasoning (NeurIPS 2023) 🧠

Welcome to the official repository for our NeurIPS 2023 paper! This README will guide you through the setup and execution of our experiments.

## 📄 Paper

- **Main Paper:** [Disentangled Counterfactual Learning for Physical Commonsense Reasoning](https://arxiv.org/pdf/2310.19559)

## 🆕 Updates

- **July 24, 2025:**  
  We have added inference code for Qwen2.5-VL, as well as code for fine-tuning Qwen2.5-VL on the PACS dataset. Additionally, we provide training and inference code for a new baseline using Qwen2.5-VL's visual encoder. More distance metrics for physical knowledge correlation have also been included.

- **February 17, 2025:**  
  We are excited to announce the release of our extended version, **Robust Disentangled Counterfactual Learning for Physical Audiovisual Commonsense Reasoning (RDCL)**, now available on arXiv. In this version, we explore scenarios involving missing modalities and introduce a new dataset based on VLM descriptions of visual information for each object.

  - **Extended Paper:** [Robust Disentangled Counterfactual Learning for Physical Audiovisual Commonsense Reasoning](https://arxiv.org/pdf/2502.12425)
  - **Dataset:** [Baidu Netdisk](https://pan.baidu.com/s/1Ei76NNkb1CFt8FJkDJDFMg) (Extraction Code: `v458`)

---

## 📥 Downloading Model Weights

To get started, download the pretrained weights for CLIP and AudioCLIP.

### CLIP

```bash
wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

### AudioCLIP

Download `AudioCLIP-Partial-Training.pt` and `bpe_simple_vocab_16e6.txt.gz` from the [AudioCLIP Releases](https://github.com/AndreyGuzhov/AudioCLIP/releases).

After downloading, place the models into the `assets` folder.

---

## 🛠️ Requirements

We recommend the following environment:

- **Python:** 3.8.10
- **PyTorch:** 1.11.0
- **CUDA:** 11.3

---

## 🏋️‍♂️ Training

### For NeurIPS 2023 Paper

#### 1. PACS Dataset

```bash
conda activate PACS
python3 train_1.py
```

#### 2. Material Classification Dataset

```bash
python3 train_classify.py
```

### For arXiv 2025 Paper

#### 1. PACS Dataset (with missing modality)

```bash
conda activate PACS
python3 train_1.py --miss_modal audio 
```

#### 2. Material Classification Dataset (with missing modality)

```bash
python3 train_classify.py --miss_modal audio 
```

---

## 🔮 Prediction

After training, you can generate predictions on the test set:

```bash
python3 predict.py -model_path PATH_TO_MODEL_WEIGHTS -split test
```

---

## 🧩 Qwen2.5-VL Integration

### 1. Constructing Qwen Inference Data from PACS

```bash
python3 PACS_data/scripts/processing_qwen_inference_multi_video_data.py
```

### 2. Inference on PACS Test Set with Qwen2.5-VL

```bash
python PACS_inference.py --model_dir Qwen/Qwen2.5-VL-3B-Instruct --tokenizer_dir Qwen/Qwen2.5-VL-3B-Instruct --split test --data_type data
```

### 3. Constructing Qwen Fine-tuning Data from PACS

```bash
python3 PACS_data/scripts/processing_qwen_finetune_data.py
```

### 4. Fine-tuning Qwen2.5-VL on PACS Train Set

```bash
cd qwen-vl-finetune
sh scripts/sft_PACS.sh
```

> We use 4 V100 GPUs for training. For parameter adjustments, please refer to the [official Qwen2.5-VL codebase](https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune).

---

## 🆕 Baseline Construction with Qwen2.5-VL Visual Encoder

Since Qwen2.5-VL inference is slow, we first extract and save the visual and image features from PACS.

### Extract Video Frame Features

```bash
python3 extract_feature.py --model_size 3B
python3 extract_feature.py --model_size 7B
python3 extract_feature.py --model_size 32B
```

### Extract Single Frame Features

```bash
python3 extract_feature_single_image.py --model_size 3B
python3 extract_feature_single_image.py --model_size 7B
python3 extract_feature_single_image.py --model_size 32B
```

### Training with Qwen2.5-VL as Baseline

```bash
python3 train_qwen_baseline.py --Qwen2_5_Size 3B
```

### Experiments with Different Distance Metrics

```bash
python3 train_qwen_baseline.py --sim_type euclidean
python3 train_qwen_baseline.py --sim_type manhattan
```

---

## 🙏 Acknowledgements

This code is adapted from:

- [AudioCLIP](https://github.com/AndreyGuzhov/AudioCLIP)
- [PACS](https://github.com/samuelyu2002/PACS)

We would like to express our gratitude to Andrey Guzhov, Samuel Yu, and all other contributors of these repositories for their invaluable work. 🙌

---

Feel free to reach out if you have any questions or need further assistance! 🚀

---