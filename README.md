# DCSHARP

3D Gaussian Splatting (3DGS) shows outstanding rendering quality for novel view synthesis. Despite its performance, the massive amount of Gaussian blobs leads to expensive run-time sorting and irregular memory access during rendering. Although 3DGS-based pruning algorithm has been widely explored, most of the current research has mainly focused on designing a proper pruning metric and the root cause behind the inevitable quality degradation remains underexplored for highly-sparse 3DGS. In particular, our investigation shows that the Spherical Harmonics (SH) of 3DGS is insufficient to capture high-frequency anisotropic reflections and specular highlights during rendering, especially with sparsified Gaussians. Motivated by that, this work proposes Direction Cosine Spherical Harmonics with Shape-Aware Pruning (DCSHARP). Specifically, the proposed Direction Cosine Spherical Harmonics (DCSH) replaces the vanilla spherical harmonics by facilitating the expressiveness of 3DGS on high-frequency and highly reflective scenes. Unlike recent works that rely on trainable masks or pseudo-rendering scores, the proposed Shape-aware Pruning method enables ``pruning on-the-fly'' while achieving high quality rendering. As a combined scheme, the proposed DCSHARP reduces the number of active Gaussians by up to 3.9x and improves rendering throughput by 1.9x with ZERO quality degradation compared to the vanilla 3DGS. Furthermore, the proposed DCSH scheme outperforms the vanilla 3DGS on all the mainstream benchmarks by simply replacing the vanilla SH with the DCSH. The source code of the proposed method will be open-sourced.


## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training Pipeline for Static and Dynamic Scenes](#SoundSpaces-training)
- [Results](#results)
- [Usage](#usage)
- [Citation](#citation)

## Installation

```bash
# Clone the repository
git clone https://github.com/Ahmedhasssan/DCSHARP.git
cd DCAHARP

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
- NeRF360
- Synthetic NeRF
- Zip-NeRF360
- Tanks & Temples
- Deep Blending
- Splatting Avatar Flame Meshes

### Data Structure

Organize your dataset in the following structure:

```
<location>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

### Training Pipeline

The preprocessing pipeline includes:

1. **Enable DCSH by re-installing**: /submodules/diff-gaussian-rasterization and /submodules/simple-knn
2. **Prepare Training Data**: Make sure to have training data in Colmap or Synthetic NeRF format
3. **Enable pruning by changing the masking filter**: Uncomment the line 522 of scene/Gaussian_model.py, you can adjust the threshold on line 459. 
   
### Usage
Note: Ensure that you provide the correct path to the original dataset. We use "--eval" flag and do not show the test data to the model during training.
```bash
export CUDA_VISIBLE_DEVICES=0
# DATA_PATH="/home/ah2288/LP_MipNerF/data/nerf_synthetic/hotdog"
python train.py -s /home/ah2288/gaussian-splatting/data/360_v2/kitchen \
    --eval \
    --checkpoint_iteration 30000 \
    --model_path "/home/ah2288/gs_baseline/gaussian-splatting/output_new/kitchen" 
```
```bash
# Simple Way
Bash train.sh
```

### Rendering only
```bash
export CUDA_VISIBLE_DEVICES=0
# DATA_PATH="/home/ah2288/LP_MipNerF/data/nerf_synthetic/hotdog"
python render.py -m "/home/ah2288/gs_baseline/gaussian-splatting/output_new/kitchen" \
    -s /home/ah2288/gaussian-splatting/data/360_v2/kitchen 
```
```bash
# Simple Way
Bash rendering.sh
```

### Full Evaluation on 3DGS pattern

```bash
export CUDA_VISIBLE_DEVICES=0
python full_eval.py -m360 "/home/ah2288/gaussian-splatting/data/360_v2" -tat "/home/ah2288/gaussian-splatting/data/tandt" -db "/home/ah2288/gaussian-splatting/data/db"

```

### Evaluation Metrics

Track the following metrics during feature distillation:

- **3D Rendering Accuracy**: PSNR, SSIM and LPIPS Scores

## Results

### Performance Comparison

|  | Bicycle | Bonsai | Counter | Kitchen | Room | Stump | Garden | Truck | Train | Johnson | Playroom |
|---|---------|--------|---------|---------|------|-------|--------|-------|-------|---------|----------|
| **SSIM** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mip-NeRF 360 | 0.693 | 0.939 | 0.895 | 0.920 | 0.913 | 0.746 | 0.816 | 0.583 | 0.632 | 0.857 | 0.660 | 0.901 | 0.900 |
| Zip-NeRF | 0.769 | 0.949 | 0.902 | 0.928 | 0.925 | 0.800 | 0.860 | 0.642 | 0.681 | - | - | 0.905 | 0.908 |
| 3DGS | 0.771 | 0.938 | 0.905 | 0.922 | 0.914 | 0.775 | 0.868 | 0.605 | 0.638 | 0.879 | 0.802 | 0.899 | 0.906 |
| Mini-Splatting | 0.798 | 0.946 | 0.913 | 0.934 | 0.928 | 0.804 | 0.878 | 0.642 | 0.640 | 0.890 | 0.817 | 0.905 | 0.908 |
| **DCSH** | **0.781** | **0.951** | **0.934** | **0.933** | **0.931** | **0.810** | **0.910** | **0.880** | **0.823** | **0.905** | **0.911** |
| **PSNR** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mip-NeRF 360 | 24.40 | 33.11 | 29.44 | 32.02 | 31.40 | 26.36 | 26.94 | 21.64 | 22.81 | 24.91 | 19.52 | 29.14 | 29.66 |
| Zip-NeRF | 25.80 | 34.46 | 29.93 | 32.50 | 32.65 | 27.55 | 28.20 | 22.40 | 23.89 | - | - | 29.32 | 30.43 |
| 3DGS | 25.25 | 31.98 | 28.70 | 30.52 | 30.63 | 26.55 | 27.41 | 21.52 | 22.49 | 25.19 | 21.10 | 28.77 | 30.04 |
| Mini-Splatting | 25.55 | 31.72 | 28.72 | 31.75 | 31.41 | 27.11 | 27.67 | 21.50 | 22.13 | 25.43 | 21.04 | 29.32 | 30.43 |
| **DCSH** | **25.84** | **32.89** | **30.47** | **31.97** | **31.89** | **26.98** | **29.07** | **25.62** | **22.69** | **29.26** | **30.38** |
| **LPIPS** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mip-NeRF 360 | 0.289 | 0.177 | 0.203 | 0.126 | 0.211 | 0.254 | 0.164 | 0.345 | 0.338 | 0.159 | 0.354 | 0.237 | 0.252 |
| Zip-NeRF | 0.208 | 0.173 | 0.185 | 0.116 | 0.196 | 0.193 | 0.118 | 0.273 | 0.242 | - | - | 0.244 | 0.243 |
| 3DGS | 0.205 | 0.205 | 0.204 | 0.129 | 0.220 | 0.210 | 0.103 | 0.336 | 0.317 | 0.148 | 0.218 | 0.244 | 0.241 |
| Mini-Splatting | 0.158 | 0.175 | 0.172 | 0.114 | 0.190 | 0.169 | 0.090 | 0.255 | 0.262 | 0.100 | 0.181 | 0.218 | 0.204 |
| **DCSH** | **0.170** | **0.085** | **0.057** | **0.064** | **0.091** | **0.150** | **0.067** | **0.064** | **0.122** | **0.102** | **0.081** |

## Citation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the contributors and the open-source community
- Special thanks to [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), which inspired this work.
- We have borrowed a lot of code from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting.git).
