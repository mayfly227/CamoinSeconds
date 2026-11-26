# CamoinSeconds

A texture optimization framework built on Stable Diffusion XL LoRA and differentiable 3D rendering, targeting visual perception robustness in autonomous driving simulation.

## Overview

CamoinSeconds places a generative model inside an end-to-end differentiable rendering loop. Images sampled from SDXL are mapped into the UV texture space of a 3D vehicle mesh, rendered into multi-view images by a differentiable renderer, and scored by an object detector whose reward is back-propagated into either the LoRA parameters or the texture representation itself.


## Features

| Module | Description |
| --- | --- |
| SDXL LoRA training | Injects LoRA into self-attention (`attn1`), cross-attention (`attn2`), or both, with configurable rank, dropout and parameter regularization |
| Differentiable rendering | Two backends, PyTorch3D and nvdiffrast, supporting UV texturing, mipmap filtering and antialiasing |
| Detector reward | Builds the reward from YOLO detections, with a margin detection loss that separates target and background confidence |
| Compositional prompt pool | Samples prompts along five axes (`material`, `color`, `pattern`, `context`, `style`), with ratio-based selection and holdout sets |
| Texture representation | Pixel-grid UV textures by default; a 2D Gaussian field with Cholesky-parameterized covariance is provided as a drop-in alternative |
| Regularization | TV smoothing, LoRA parameter regularization, CLIP semantic preservation, diversity repulsion and trajectory anchoring |
| Scene augmentation | Differentiable random augmentations: affine, color jitter, Gaussian noise, motion blur, fog, rain and snow |

## Repository layout

```
.
├── train_lora_sdxl.py         Main SDXL LoRA training script
├── train_lora_sdxlexp.py      Experimental branch with a slightly reduced argument set
├── prompt_pool.py             Compositional prompt pool
├── data/
│   ├── __init__.py            Camouflage texture prompt library, grouped by scene
│   └── bgdataset.py           Background image dataset
├── meshrender/
│   ├── __init__.py            PyTorch3D rendering backend
│   └── nvdiffrast_uv.py       nvdiffrast UV rendering backend
├── utils/
│   ├── loss.py                Loss functions
│   ├── gaussian_texture.py    2D Gaussian UV texture representation
│   ├── gaussian_image.py      Gaussian image and Cholesky parameterization
│   ├── aug.py                 Random augmentation suite (ROA) and differentiable fog
│   ├── yolo.py                YOLO reward model wrapper
│   ├── cliputil.py            CLIP text-image similarity
│   ├── core.py                GPU selection, UV mask loading, tensor conversion
│   └── dataset.py             Prompt-image mapping dataset
└── LICENSE
```

## Requirements

Core dependencies:

```
python >= 3.10
torch >= 2.1
diffusers
transformers
accelerate
peft
numpy
pillow
tqdm
```

Optional rendering backends, install as needed:

- `pytorch3d` — required by `meshrender/__init__.py`
- `nvdiffrast` — required by `meshrender/nvdiffrast_uv.py`

Detector dependency:

```
ultralytics
```

## Getting started

### 1. Prepare model weights

Download the following models locally in advance:

- SDXL base 1.0 (`stabilityai/stable-diffusion-xl-base-1.0`)
- CLIP ViT-L/14 (`openai/clip-vit-large-patch14`)
- A YOLO detector checkpoint, for example `yolov3u.pt`

### 2. Prepare mesh and texture assets

Vehicle meshes and UV masks must be prepared for your own scene and passed through the corresponding arguments:

- `--mesh_obj_path`: path to the OBJ mesh
- `--uv_mask_path`: UV validity mask
- `--texture_image_size`: texture resolution, 640 by default
- `--use_carla_mesh_transform`: apply the CARLA mesh coordinate-system fix

### 3. Launch training

```bash
python train_lora_sdxlexp.py \
    --pretrained_model_name_or_path /path/to/stable-diffusion-xl-base-1.0 \
    --clip_path /path/to/clip-vit-large-patch14 \
    --reward_model_path yolomodel/yolov3u.pt \
    --output_dir output/run1 \
    --rank 64 \
    --lora_scope both \
    --num_reward_views 8 \
    --bptt 19 \
    --max_train_steps 3000 \
    --use_prompt_pool \
    --clip_weight 0.3 \
    --reward_weight 1.0
```

## Training arguments

| Argument | Description |
| --- | --- |
| `--output_dir` | Output directory for LoRA weights, logs and rendered results |
| `--pretrained_model_name_or_path` | Path to the SDXL base model |
| `--clip_path` | Path to the CLIP model used for semantic preservation |
| `--reward_model_path` | Path to the detector checkpoint |
| `--rank` | LoRA rank, commonly 32 or 64 |
| `--lora_scope` | LoRA injection target: `attn1` (self-attention), `attn2` (cross-attention) or `both` |
| `--num_reward_views` | Number of rendered views per optimization step |
| `--bptt` | Truncated back-propagation depth through diffusion steps; higher values cost more memory |
| `--max_train_steps` | Total number of training steps |
| `--use_prompt_pool` | Enable compositional prompt-pool sampling |
| `--clip_weight` | Weight of the CLIP semantic preservation loss |
| `--reward_weight` | Weight of the detector reward |
| `--lora_reg_weight` | Weight of LoRA parameter regularization |
| `--gradient_accumulation_steps` | Gradient accumulation steps for low-memory setups |
| `--gradient_checkpointing` | Enable gradient checkpointing to further reduce memory |
| `--texture_image_size` | UV texture resolution |

## License

This project is released under the [GNU General Public License v3.0](LICENSE).
