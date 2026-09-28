# DCSHARP

3D Gaussian Splatting (3DGS) shows outstanding rendering quality for novel view synthesis. Despite its performance, the massive amount of Gaussian blobs leads to expensive run-time sorting and irregular memory access during rendering. Although 3DGS-based pruning has been widely explored, most of the current research has focused on designing a pruning metric, and the reason quality drops for highly sparse 3DGS is still underexplored. In particular, spherical harmonics (SH) in 3DGS struggle to capture high-frequency anisotropic reflections and specular highlights once the Gaussians are sparsified.

Direction Cosine Spherical Harmonics with Shape-Aware Pruning (DCSHARP) addresses that gap. Direction Cosine Spherical Harmonics (DCSH) replace the vanilla SH basis so each Gaussian can express high-frequency, view-dependent color. Shape-aware pruning removes Gaussians during training, using the scale-gradient magnitude, without a learned mask or a pseudo-rendering score. Together, the method reduces the number of active Gaussians by up to 3.9x and raises rendering throughput by 1.9x with no quality drop against vanilla 3DGS. DCSH alone, with pruning left off, also beats vanilla 3DGS on the standard benchmarks.

## Table of Contents

- [Installation](#installation)
- [Docker](#docker)
- [Dataset preparation](#dataset-preparation)
- [Training](#training)
- [Rendering and evaluation](#rendering-and-evaluation)
- [Shape-aware pruning](#shape-aware-pruning)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Installation

The tested environment is the Docker image below: Ubuntu 22.04, CUDA 11.8, Python 3.10, and PyTorch 2.1.2. A CUDA GPU is required to train and render. `environment.yml` is the original 3DGS conda pin (Python 3.7, PyTorch 1.12, CUDA 11.6) and is not the setup this README tests.

DCSH spherical harmonics are compiled into `submodules/diff-gaussian-rasterization-new`. That is the rasterizer `train.py` and `render.py` import as `diff_gaussian_rasterization`. The matching nearest-neighbor extension is `submodules/simple-knn-v1`. The directories `submodules/diff-gaussian-rasterization` and `submodules/simple-knn` are empty upstream slots; installing them builds vanilla 3DGS. `environment.yml` installs the DCSH trees.

With the default flags, both training and rendering evaluate spherical harmonics inside that CUDA extension (`convert_SHs_python` stays off). The extension exports `sh_variant() == "dcsh"`. `gaussian_renderer` checks this on import and stops if a vanilla rasterizer was installed instead. Do not leave a prebuilt `_C*.so` in the package directory; those binaries are ignored by git and the image always compiles the current CUDA sources.

GLM headers are not included in the repository. The rasterizer build expects them at `submodules/diff-gaussian-rasterization-new/third_party/glm`.

```bash
git clone https://github.com/Ahmedhasssan/DCSHARP.git
cd DCSHARP

python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install "numpy==1.26.4" "plyfile==1.0.3" "opencv-python-headless==4.10.0.84" \
    tqdm imageio matplotlib joblib scipy lpips tensorboard
python -m pip install kiui --no-deps
python -m pip install lazy_loader varname objprint rich questionary pyyaml pillow trimesh
python docker/patch_kiui.py

git clone --depth 1 --branch 0.9.9.8 https://github.com/g-truc/glm.git \
    submodules/diff-gaussian-rasterization-new/third_party/glm
python -m pip uninstall -y diff_gaussian_rasterization simple_knn || true
python -m pip install --no-build-isolation --force-reinstall --no-cache-dir submodules/simple-knn-v1
python -m pip install --no-build-isolation --force-reinstall --no-cache-dir submodules/diff-gaussian-rasterization-new
python -c "from diff_gaussian_rasterization import _C; assert _C.sh_variant() == 'dcsh'"
```

`docker/patch_kiui.py` fills `kiui/typing.py`. Release 0.3.5 ships that file empty, and `train.py` imports `kiui.cam` before it parses arguments.

`train.py` imports `lpips`, `kiui`, and `imageio` at startup, so those packages are required even for a normal training run.

## Docker

The image compiles both CUDA extensions and can train one scene or run the full Mip-NeRF 360 / Tanks & Temples / Deep Blending evaluation. The host needs Docker, an NVIDIA driver, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

`TORCH_CUDA_ARCH_LIST` defaults to `7.5;8.0;8.6;8.9;9.0` (T4 / Turing, A100, RTX 30, RTX 40 / A6000, H100). Pass a shorter list to speed up the compile:

```bash
docker build --network=host -t dcsharp:latest .
docker build --network=host -t dcsharp:latest --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6" .
```

`--network=host` lets the build reach the host DNS resolver. Without it, `apt-get` inside the CUDA image often fails with `Temporary failure resolving`.

Check that the image starts and the extensions import:

```bash
docker run --rm dcsharp:latest help
docker run --rm --gpus all dcsharp:latest \
    python -c "import torch, diff_gaussian_rasterization, simple_knn; print(torch.__version__, torch.cuda.is_available())"
```

Train one scene. `DATA_DIR` is a COLMAP scene (with `sparse/` and `images/`) or a Blender scene (with `transforms_train.json`). Checkpoints and logs go to `OUTPUT_DIR`.

```bash
export DATA_DIR=/path/to/kitchen
export OUTPUT_DIR=/path/to/output/kitchen
docker compose run --rm train
```

The same image renders a trained model and scores it:

```bash
docker compose run --rm render
docker compose run --rm metrics
```

Full evaluation trains every scene, renders iterations 7000 and 30000, then writes PSNR, SSIM, and LPIPS.

```bash
export M360_PATH=/path/to/360_v2
export TAT_PATH=/path/to/tandt
export DB_PATH=/path/to/db
export OUTPUT_DIR=/path/to/eval
docker compose run --rm eval
```

Equivalent `docker run` commands, without Compose:

```bash
docker run --rm --gpus all \
    -v "$DATA_DIR":/data \
    -v "$OUTPUT_DIR":/output \
    dcsharp:latest train -s /data --eval -m /output --checkpoint_iterations 30000

docker run --rm --gpus all \
    -v "$DATA_DIR":/data \
    -v "$OUTPUT_DIR":/output \
    dcsharp:latest render -m /output -s /data

docker run --rm --gpus all \
    -v "$OUTPUT_DIR":/output \
    dcsharp:latest metrics -m /output

docker run --rm --gpus all \
    -v "$M360_PATH":/data/360 \
    -v "$TAT_PATH":/data/tandt \
    -v "$DB_PATH":/data/db \
    -v "$OUTPUT_DIR":/output \
    dcsharp:latest eval -m360 /data/360 -tat /data/tandt -db /data/db --output_path /output
```

The scene directory is mounted writable. The first training run writes `sparse/0/points3D.ply` from `points3D.bin`.

`eval` accepts `--skip_training`, `--skip_rendering`, and `--skip_metrics` when you only want part of that pipeline.

## Dataset preparation

`scene/__init__.py` recognizes three layouts:

| Layout | How it is detected | Typical datasets |
| --- | --- | --- |
| COLMAP | a `sparse/` directory | Mip-NeRF 360, Tanks & Temples, Deep Blending, Zip-NeRF after COLMAP conversion |
| Blender | `transforms_train.json` | Synthetic NeRF |
| VR-NeRF | an `images-jpeg-1k` directory | VR-NeRF captures |

COLMAP scenes use this structure:

```
<scene>
|-- images
|   |-- <image 0>
|   `-- <image 1>
`-- sparse
    `-- 0
        |-- cameras.bin
        |-- images.bin
        `-- points3D.bin
```

`--eval` holds out every 8th COLMAP image for testing (`llffhold=8`) and does not train on that split. Use `convert.py` when the images still need COLMAP undistortion into this layout.

Download the public Mip-NeRF 360 and Tanks & Temples / Deep Blending archives, then point the scripts at the extracted scene roots:

```bash
mkdir -p data && cd data
wget -O 360_v2.zip https://storage.googleapis.com/gresearch/refraw360/360_v2.zip
wget -O tandt_db.zip https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
unzip -q -o 360_v2.zip
unzip -q -o tandt_db.zip
```

The Mip-NeRF 360 zip extracts each scene next to the archive. Tanks & Temples and Deep Blending extract into `tandt/` and `db/`. Flowers and treehill are not in the public archive.

```bash
export M360_PATH=/path/to/data
export TAT_PATH=/path/to/data/tandt
export DB_PATH=/path/to/data/db
```

`full_eval.py` then reads:

```
data/{bicycle,garden,stump,room,counter,kitchen,bonsai}
data/tandt/{truck,train}
data/db/{drjohnson,playroom}
```

## Training

Training runs for 30,000 iterations. The model is tested and saved at iterations 7,000 and 30,000. A checkpoint is written at iteration 30,000. The flag is `--checkpoint_iterations` (plural).

```bash
export CUDA_VISIBLE_DEVICES=0
export DATA_PATH=/path/to/kitchen
export MODEL_PATH=/path/to/output/kitchen
bash train.sh
```

`bash train.sh` forwards any extra arguments to `train.py`. The direct command is:

```bash
python train.py -s "$DATA_PATH" \
    --eval \
    --checkpoint_iterations 30000 \
    -m "$MODEL_PATH"
```

Logged metrics during the test iterations are L1, PSNR, SSIM, and LPIPS.

## Rendering and evaluation

Render the train and test splits of a trained scene, then score the test renders:

```bash
export DATA_PATH=/path/to/kitchen
export MODEL_PATH=/path/to/output/kitchen
bash rendering.sh
python metrics.py -m "$MODEL_PATH"
```

`metrics.py` writes `results.json` and `per_view.json` under the model path. It reports PSNR, SSIM, and LPIPS.

The benchmark driver trains, renders, and scores every scene:

```bash
export M360_PATH=/path/to/360_v2
export TAT_PATH=/path/to/tandt
export DB_PATH=/path/to/db
export OUTPUT_PATH=/path/to/eval
bash full_eval.sh
```

Training time for each dataset group is written to `$OUTPUT_PATH/timing.txt`.

## Shape-aware pruning

Pruning is already enabled. In `scene/gaussian_model.py`, `densify_and_prune` ranks Gaussians by scale-gradient magnitude and marks the top fraction for removal. That fraction is `0.02` in the `torch.topk` call (about line 459). The mask is applied on the following line, which is already uncommented:

```python
prune_mask = torch.logical_or(prune_mask, close_mask.squeeze())  # pruning on
```

Change `0.02` to keep a different fraction of high scale-gradient Gaussians. Comment that `logical_or` line to train with DCSH and the standard 3DGS opacity prune only.

## Results

The tables below are the numbers previously listed in this README, with each value placed under the scene it belongs to. Baseline rows had two extra entries (Flowers and Treehill) that the old header omitted, so those columns no longer sit under Truck and Train. DCSH has no Flowers or Treehill number in this repository; those cells are left blank.

SSIM and PSNR are higher-is-better. LPIPS is lower-is-better.

### Mip-NeRF 360

| Method | Metric | Bicycle | Bonsai | Counter | Kitchen | Room | Stump | Garden | Flowers | Treehill |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mip-NeRF 360 | SSIM | 0.693 | 0.939 | 0.895 | 0.920 | 0.913 | 0.746 | 0.816 | 0.583 | 0.632 |
| Zip-NeRF | SSIM | 0.769 | 0.949 | 0.902 | 0.928 | 0.925 | 0.800 | 0.860 | 0.642 | 0.681 |
| 3DGS | SSIM | 0.771 | 0.938 | 0.905 | 0.922 | 0.914 | 0.775 | 0.868 | 0.605 | 0.638 |
| Mini-Splatting | SSIM | 0.798 | 0.946 | 0.913 | 0.934 | 0.928 | 0.804 | 0.878 | 0.642 | 0.640 |
| **DCSH** | **SSIM** | **0.781** | **0.951** | **0.934** | **0.933** | **0.931** | **0.810** | **0.910** |  |  |
| mip-NeRF 360 | PSNR | 24.40 | 33.11 | 29.44 | 32.02 | 31.40 | 26.36 | 26.94 | 21.64 | 22.81 |
| Zip-NeRF | PSNR | 25.80 | 34.46 | 29.93 | 32.50 | 32.65 | 27.55 | 28.20 | 22.40 | 23.89 |
| 3DGS | PSNR | 25.25 | 31.98 | 28.70 | 30.52 | 30.63 | 26.55 | 27.41 | 21.52 | 22.49 |
| Mini-Splatting | PSNR | 25.55 | 31.72 | 28.72 | 31.75 | 31.41 | 27.11 | 27.67 | 21.50 | 22.13 |
| **DCSH** | **PSNR** | **25.84** | **32.89** | **30.47** | **31.97** | **31.89** | **26.98** | **29.07** |  |  |
| mip-NeRF 360 | LPIPS | 0.289 | 0.177 | 0.203 | 0.126 | 0.211 | 0.254 | 0.164 | 0.345 | 0.338 |
| Zip-NeRF | LPIPS | 0.208 | 0.173 | 0.185 | 0.116 | 0.196 | 0.193 | 0.118 | 0.273 | 0.242 |
| 3DGS | LPIPS | 0.205 | 0.205 | 0.204 | 0.129 | 0.220 | 0.210 | 0.103 | 0.336 | 0.317 |
| Mini-Splatting | LPIPS | 0.158 | 0.175 | 0.172 | 0.114 | 0.190 | 0.169 | 0.090 | 0.255 | 0.262 |
| **DCSH** | **LPIPS** | **0.170** | **0.085** | **0.057** | **0.064** | **0.091** | **0.150** | **0.067** |  |  |

### Tanks & Temples

| Method | Metric | Truck | Train |
| --- | --- | --- | --- |
| mip-NeRF 360 | SSIM | 0.857 | 0.660 |
| Zip-NeRF | SSIM | — | — |
| 3DGS | SSIM | 0.879 | 0.802 |
| Mini-Splatting | SSIM | 0.890 | 0.817 |
| **DCSH** | **SSIM** | **0.880** | **0.823** |
| mip-NeRF 360 | PSNR | 24.91 | 19.52 |
| Zip-NeRF | PSNR | — | — |
| 3DGS | PSNR | 25.19 | 21.10 |
| Mini-Splatting | PSNR | 25.43 | 21.04 |
| **DCSH** | **PSNR** | **25.62** | **22.69** |
| mip-NeRF 360 | LPIPS | 0.159 | 0.354 |
| Zip-NeRF | LPIPS | — | — |
| 3DGS | LPIPS | 0.148 | 0.218 |
| Mini-Splatting | LPIPS | 0.100 | 0.181 |
| **DCSH** | **LPIPS** | **0.064** | **0.122** |

### Deep Blending

| Method | Metric | DrJohnson | Playroom |
| --- | --- | --- | --- |
| mip-NeRF 360 | SSIM | 0.901 | 0.900 |
| Zip-NeRF | SSIM | 0.905 | 0.908 |
| 3DGS | SSIM | 0.899 | 0.906 |
| Mini-Splatting | SSIM | 0.905 | 0.908 |
| **DCSH** | **SSIM** | **0.905** | **0.911** |
| mip-NeRF 360 | PSNR | 29.14 | 29.66 |
| Zip-NeRF | PSNR | 29.32 | 30.43 |
| 3DGS | PSNR | 28.77 | 30.04 |
| Mini-Splatting | PSNR | 29.32 | 30.43 |
| **DCSH** | **PSNR** | **29.26** | **30.38** |
| mip-NeRF 360 | LPIPS | 0.237 | 0.252 |
| Zip-NeRF | LPIPS | 0.244 | 0.243 |
| 3DGS | LPIPS | 0.244 | 0.241 |
| Mini-Splatting | LPIPS | 0.218 | 0.204 |
| **DCSH** | **LPIPS** | **0.102** | **0.081** |

## Citation

Please cite the original 3D Gaussian Splatting paper, which this code extends:

```bibtex
@article{kerbl3Dgaussians,
  author = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal = {ACM Transactions on Graphics},
  number = {4},
  volume = {42},
  month = {July},
  year = {2023}
}
```

## License

This project is distributed under the Inria and MPII Gaussian Splatting research license in [LICENSE.md](LICENSE.md). That license permits non-commercial research and evaluation use. It is not the MIT license.

## Acknowledgments

- The implementation is built on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) from Inria GRAPHDECO.
- Project page for the original method: [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).
