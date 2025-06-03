# DCSHARP

3D Gaussian Splatting (3DGS) shows outstanding rendering quality for novel view synthesis. Despite its performance, the massive amount of Gaussian blobs leads to expensive run-time sorting and irregular memory access during rendering. Although 3DGS-based pruning algorithm has been widely explored, most of the current research has mainly focused on designing a proper pruning metric and the root cause behind the inevitable quality degradation remains underexplored for highly-sparse 3DGS. In particular, our investigation shows that the Spherical Harmonics (SH) of 3DGS is insufficient to capture high-frequency anisotropic reflections and specular highlights during rendering, especially with sparsified Gaussians. Motivated by that, this work proposes Direction Cosine Spherical Harmonics with Shape-Aware Pruning (DCSHARP). Specifically, the proposed Direction Cosine Spherical Harmonics (DCSH) replaces the vanilla spherical harmonics by facilitating the expressiveness of 3DGS on high-frequency and highly reflective scenes. Unlike recent works that rely on trainable masks or pseudo-rendering scores, the proposed Shape-aware Pruning method enables ``pruning on-the-fly'' while achieving high quality rendering. As a combined scheme, the proposed DCSHARP reduces the number of active Gaussians by up to 3.9x and improves rendering throughput by 1.9x with ZERO quality degradation compared to the vanilla 3DGS. Furthermore, the proposed DCSH scheme outperforms the vanilla 3DGS on all the mainstream benchmarks by simply replacing the vanilla SH with the DCSH. The source code of the proposed method will be open-sourced.


## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training Pipeline for SoundSpaces Dataset](#SoundSpaces-training)
- [Results](#results)
- [Usage](#usage)
- [Citation](#citation)

## Installation

```bash
# Clone the repository
git clone https://github.com/Ahmedhasssan/SAVAF-AV.git
cd SAVAF-AV

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python >= 3.10.0
- PyTorch >= 2.6.0
- torchvision >= 0.20.0
- numpy
- opencv-python
- PIL

## Dataset Preparation

### Supported Datasets

This project supports the following datasets:
- SoundSpaces and NVS-Replay

### Data Structure

Organize your dataset in the following structure:

```
├── 1
│   ├── binaural_syn_re.wav
│   ├── feats_train.pkl
│   ├── feats_val.pkl
│   ├── frames
│   │   ├── 00001.png
|   |   ├── ...
│   │   ├── 00616.png
│   ├── source_syn_re.wav
│   ├── transforms_scale_train.json
│   ├── transforms_scale_val.json
│   ├── transforms_train.json
│   └── transforms_val.json
├── ...
├── 13
└── position.json
```

### Training Pipeline for SoundSpaces Dataset

The preprocessing pipeline includes:

1. **Audio Visual Rendering**: Train the 3DGS for visual rendering on RGB data and save the sparse Gaussians for Audio learning.
2. **Audio Synthesis**: Load the locally stored Gaussians and implement binaural audio synthesis using Multihead Acoustic Field Attention Network

### Usage
Note: Ensure that you provide the correct path to the original dataset.
```bash
export CUDA_VISIBLE_DEVICES=2,3,4,5,6
python nvas/trainer_v1.py \
    --version soundspaces_nvas \
    --model MHAFAN \
    --dataset soundspaces_nvas \
    --dataset-dir data/synthetic_dataset/v16 \
    --num-channel 2 \
    --n-gpus 1 \
    --num-node 16 \
    --num-worker 4 \
    --batch-size 12 \
    --gpu-mem32 \
    --decode-wav \
    --max-epochs 100 \
    --use-tgt-pose \
    --use-tgt-rotation \
    --encode-sincos \
    --mag \
    --one-speaker \
    --auto-resume \
    --audio-len 16000 \
    --remove-delay \
    --highpass-filter \
    --use-rgb \
    --remove-hyperconv \
    --acausal \
    --progress_bar \
    --use-speaker-bboxes \
    --metadata-file 'cleaned_metadata_v3.json'
```
```bash
# Simple Way
Bash train.sh and remove "visualization, test, and eval-best"
```

### Inference and visualization only
Provide the av_checkpoints and use flag --eval_aud
```bash
export CUDA_VISIBLE_DEVICES=2,3,4,5,6
python nvas/trainer_v1.py \
    --version soundspaces_nvas \
    --model MHAFAN \
    --dataset soundspaces_nvas \
    --dataset-dir data/synthetic_dataset/v16 \
    --num-channel 2 \
    --n-gpus 1 \
    --num-node 16 \
    --num-worker 4 \
    --batch-size 12 \
    --gpu-mem32 \
    --decode-wav \
    --max-epochs 100 \
    --use-tgt-pose \
    --use-tgt-rotation \
    --encode-sincos \
    --mag \
    --one-speaker \
    --auto-resume \
    --audio-len 16000 \
    --remove-delay \
    --highpass-filter \
    --use-rgb \
    --remove-hyperconv \
    --acausal \
    --progress_bar \
    --use-speaker-bboxes \
    --metadata-file 'cleaned_metadata_v3.json' \
    --visualize \
    --test \
    --eval-best
```
```bash
# Simple Way
Bash train.sh with "--visualization, --test, and --eval-best"
```
### Evaluation Metrics

Track the following metrics during feature distillation:

- **3D Generation Accuracy**: PSNR, SSIM and LPIPS Scores
- **Audio Synthesis Quality**: MAG distance, ENV distance, EDT, T60 and C50
- **Audio Synthesis Resource Utilization**: Memory and FPS

## Results

### Performance Comparison

| Methods | Audio | Visual | T60 (%) ↓ | C50 (dB) ↓ | EDT (sec) ↓ | Memory (MB) ↓ | FPS ↑ |
|---------|-------|--------|-----------|------------|-------------|---------------|-------|
| Opus-nearest | ✓ | ✗ | 10.10 | 3.58 | 0.115 | - | - |
| Opus-linear | ✓ | ✗ | 8.64 | 3.13 | 0.097 | - | - |
| AAC-nearest | ✓ | ✗ | 9.35 | 1.67 | 0.059 | - | - |
| AAC-linear | ✓ | ✗ | 7.88 | 1.68 | 0.057 | - | - |
| INRAS | ✓ | ✗ | 3.14 | 0.60 | 0.019 | 1.24 | 180 |
| NAF | ✓ | ✗ | 3.18 | 1.06 | 0.031 | 1.10 | 99 |
| AV-NeRF | ✓ | ✓ | 2.47 | 0.57 | 0.016 | 48 | 79 |
| AV-GS | ✓ | ✓ | 2.23 | 0.53 | 0.014 | 18.40 | 12.5 |
| **SAVAF** | ✓ | ✓ | **2.20** | **0.45** | **0.023** | **5.65** | **115** |

## Citation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the contributors and the open-source community
- Special thanks to [NVS](https://arxiv.org/abs/2301.08730), which inspired this work.
- We have borrowed some code from [NVS](https://github.com/facebookresearch/novel-view-acoustic-synthesis) for data loader preparation and baseline.
