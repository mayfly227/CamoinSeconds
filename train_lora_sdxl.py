import os
from datetime import datetime

from data.bgdataset import get_background_dataloader
from utils import get_prompts_by_speed
from utils.core import get_free_gpu, load_uv_mask, pil_from_tensor
from utils.loss import lora_regularization
from utils.validation import TestASR, TestASRQuick

try:
    get_free_gpu()
except Exception as e:
    print(
        f"Warning: get_free_gpu() failed with error: {e}. Proceeding without setting CUDA_VISIBLE_DEVICES."
    )
    exit(0)
# if len(sys.argv) > 1:
#     print("传了命令行参数")
#     if len(sys.argv) < 2:
#         os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     print("参数列表:", sys.argv[1:])  # 排除脚本名本身
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# else:
#     print("没有传任何命令行参数")
#     os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from transformers import CLIPModel, CLIPProcessor

from meshrender import TextureRenderModel, tensor_rgb_to_pil
from utils.cliputil import clip_text_image_similarity
from utils.yolo import YoloRewardModel

try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for `train_reward_lora_yolo.py`. Install `torch` first."
    ) from exc

try:
    import transformers
    from transformers import AutoTokenizer, PretrainedConfig
except ImportError as exc:
    raise ImportError(
        "Transformers is required for `train_reward_lora_yolo.py`. Install `transformers` first."
    ) from exc

try:
    import diffusers
    from diffusers import (
        AutoencoderKL,  # pyright: ignore[reportPrivateImportUsage]
        DDIMScheduler,  # pyright: ignore[reportPrivateImportUsage]
        DDPMScheduler,  # pyright: ignore[reportPrivateImportUsage]
        DPMSolverMultistepScheduler,  # pyright: ignore[reportPrivateImportUsage]
        StableDiffusionXLPipeline,  # pyright: ignore[reportPrivateImportUsage]
        UNet2DConditionModel,  # pyright: ignore[reportPrivateImportUsage]
    )
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import cast_training_params, free_memory
    from diffusers.utils import (
        check_min_version,
        convert_state_dict_to_diffusers,  # pyright: ignore[reportPrivateImportUsage]
        is_wandb_available,  # pyright: ignore[reportPrivateImportUsage]
    )
    from diffusers.utils.import_utils import is_xformers_available
    from diffusers.utils.torch_utils import is_compiled_module
except ImportError as exc:
    raise ImportError(
        "Diffusers is required for `train_reward_lora_yolo.py`. Install `diffusers` first."
    ) from exc

try:
    from pytorch3d.renderer import TexturesUV, camera_position_from_spherical_angles
except ImportError as exc:
    raise ImportError(
        "PyTorch3D and the local `meshrender` helpers are required for this script."
    ) from exc

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image

if is_wandb_available():
    import wandb


check_min_version("0.37")
logger = get_logger(__name__)

LORA_TARGET_LEAF_MODULES = (
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "add_k_proj",
    "add_v_proj",
)

DEFAULT_LIGHT_LOCATION = (0.0, 5.0, -3.0)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str,
    revision: Optional[str],
    subfolder: str = "text_encoder",
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    raise ValueError(f"{model_class} is not supported.")


def tokenize_prompt(
    tokenizer, prompt: Sequence[str], tokenizer_max_length: Optional[int] = None
):
    max_length = (
        tokenizer_max_length
        if tokenizer_max_length is not None
        else tokenizer.model_max_length
    )
    return tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )


def encode_prompt_with_text_encoder(
    text_encoder,
    input_ids,
    attention_mask,
    text_encoder_use_attention_mask: bool = False,
):
    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None
    text_encoder_output = text_encoder(
        input_ids.to(text_encoder.device),
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=False,
    )
    pooled_prompt_embeds = text_encoder_output[0]
    prompt_embeds = text_encoder_output[-1][-2]
    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reward-driven LoRA fine-tuning with UV masking, PyTorch3D render, and YOLO reward."
    )
    # 预训练扩散模型路径
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/home/dk/data/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b",
    )
    # 模型版本号或分支名
    parser.add_argument("--revision", type=str, default=None)
    # 模型权重变体，例如 fp16
    parser.add_argument("--variant", type=str, default=None)
    # 单独指定 tokenizer 路径
    parser.add_argument("--tokenizer_name", type=str, default=None)
    # 训练输出目录
    parser.add_argument("--output_dir", type=str, default="rewardyolo")
    # 日志子目录
    parser.add_argument("--logging_dir", type=str, default="logs")
    # 随机种子
    parser.add_argument("--seed", type=int, default=42)
    # 训练日志上报后端
    parser.add_argument("--report_to", type=str, default="tensorboard")
    # 混合精度训练模式
    parser.add_argument(
        "--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"]
    )
    # 是否启用 TF32 加速
    parser.add_argument("--allow_tf32", action="store_true")
    # 梯度累积步数
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # 是否启用梯度检查点节省显存
    parser.add_argument("--gradient_checkpointing", default=False, type=bool)
    # 是否启用 xFormers 高效注意力
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", type=bool, default=False
    )
    # LoRA 秩
    parser.add_argument("--rank", type=int, default=64)
    # LoRA dropout 概率
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # LoRA 注入范围
    parser.add_argument(
        "--lora_scope",
        type=str,
        default="both",
        choices=["attn1", "attn2", "both"],
        help="Choose whether LoRA is injected into self-attention, cross-attention, or both.",
    )
    #
    parser.add_argument("--clip_weight", type=float, default=0.2)
    # 学习率
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    # AdamW 的 beta1
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    # AdamW 的 beta2
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    # AdamW 权重衰减
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    # AdamW 数值稳定项
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    # 梯度裁剪上限
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    # 学习率调度器类型
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    # 学习率预热步数
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    # 学习率循环次数
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    # 多项式调度的幂指数
    parser.add_argument("--lr_power", type=float, default=1.0)
    # 总训练步数
    parser.add_argument("--max_train_steps", type=int, default=1000)
    # 每隔多少步保存 checkpoint
    parser.add_argument("--checkpointing_steps", type=int, default=50)
    # 最多保留多少个 checkpoint
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    # 每隔多少步做一次验证
    parser.add_argument("--validation_steps", type=int, default=100)
    # reward 分支的 batch 大小
    parser.add_argument("--reward_batch_size", type=int, default=1)
    # 每步用于平均 reward 的渲染视角数量
    parser.add_argument("--num_reward_views", type=int, default=1)
    # 扩散采样步数
    parser.add_argument("--num_inference_steps", type=int, default=20)
    # CFG 引导系数
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    # 扩散生成分辨率
    parser.add_argument("--resolution", type=int, default=512)
    # 渲染图像分辨率
    parser.add_argument("--render_image_size", type=int, default=640)
    # UV 纹理图分辨率
    parser.add_argument("--texture_image_size", type=int, default=640)
    # tokenizer 最大长度
    parser.add_argument("--tokenizer_max_length", type=int, default=None)
    # 文本编码时是否传 attention mask
    parser.add_argument("--text_encoder_use_attention_mask", action="store_true")
    # 触发词
    parser.add_argument("--trigger_token", type=str, default="TQL")
    # 目标类别名称
    parser.add_argument("--target_class_name", type=str, default="Camouflage texture")
    # 目标类别 ID
    parser.add_argument("--target_class_id", type=int, default=None)
    # YOLO 奖励模型路径
    parser.add_argument("--reward_model_path", type=str, default="yolomodel/yolo12m.pt")
    # reward 项的损失权重
    parser.add_argument("--reward_weight", type=float, default=1.0)
    # LoRA 正则项权重
    parser.add_argument("--lora_reg_weight", type=float, default=1e-4)
    # 负面提示词
    parser.add_argument("--negative_prompt", type=str, default="")
    # prompt 模板文件路径
    parser.add_argument("--prompt_templates_file", type=str, default=None)
    # 验证时使用的 prompt 列表
    parser.add_argument("--eval_prompts", nargs="*", default=None)
    # 训练提示词采样范围，如: --prompt bio:0.2 arc:0.2 或 --prompt all:0.2
    parser.add_argument(
        "--prompt",
        nargs="*",
        default=["all:1.0"],
        help=(
            "Prompt subset spec. Examples: '--prompt bio:0.2 arc:0.2' or "
            "'--prompt all:0.2'. Ratio must be in [0, 1]."
        ),
    )
    # 渲染所用 OBJ 模型路径
    parser.add_argument("--mesh_obj_path", type=str, default="pytorch3d_Etron.obj")
    # UV 掩码图片路径
    parser.add_argument("--uv_mask_path", type=str, default="img_optim.png")
    # UV 掩码外区域的填充值
    parser.add_argument("--background_fill_value", type=float, default=0.0)
    # 相机采样策略
    parser.add_argument(
        "--camera_sampler",
        type=str,
        default="legacy",
        choices=["legacy", "stratified"],
    )
    # 方位角分桶数量
    parser.add_argument("--camera_azimuth_bins", type=int, default=12)
    # 方位角桶内抖动范围
    parser.add_argument("--camera_azimuth_jitter", type=float, default=10.0)
    # 相机俯仰角最小值
    parser.add_argument("--camera_elevation_min", type=float, default=-5.0)
    # 相机俯仰角最大值
    parser.add_argument("--camera_elevation_max", type=float, default=25.0)
    # 相机俯仰角采样峰值
    parser.add_argument("--camera_elevation_mode", type=float, default=10.0)
    # 相机距离最小值
    parser.add_argument("--camera_distance_min", type=float, default=7.0)
    # 相机距离最大值
    parser.add_argument("--camera_distance_max", type=float, default=9.5)
    # 相机距离采样峰值
    parser.add_argument("--camera_distance_mode", type=float, default=8.0)
    # 是否启用灯光随机化
    parser.add_argument("--randomize_lighting", action="store_true")
    # 灯光方位角最小值
    parser.add_argument("--light_azimuth_min", type=float, default=0.0)
    # 灯光方位角最大值
    parser.add_argument("--light_azimuth_max", type=float, default=360.0)
    # 灯光俯仰角最小值
    parser.add_argument("--light_elevation_min", type=float, default=20.0)
    # 灯光俯仰角最大值
    parser.add_argument("--light_elevation_max", type=float, default=75.0)
    # 灯光俯仰角采样峰值
    parser.add_argument("--light_elevation_mode", type=float, default=45.0)
    # 灯光距离最小值
    parser.add_argument("--light_distance_min", type=float, default=4.5)
    # 灯光距离最大值
    parser.add_argument("--light_distance_max", type=float, default=7.0)
    # 灯光距离采样峰值
    parser.add_argument("--light_distance_mode", type=float, default=5.5)
    args = parser.parse_args()

    if args.reward_batch_size != 1:
        raise ValueError(
            "v1 only supports `reward_batch_size=1` because the PyTorch3D render path is single-texture."
        )
    if args.num_reward_views < 1:
        raise ValueError("`num_reward_views` must be positive.")
    if args.max_train_steps < 1:
        raise ValueError("`max_train_steps` must be positive.")
    if args.num_inference_steps < 1:
        raise ValueError("`num_inference_steps` must be positive.")
    if args.resolution % 8 != 0:
        raise ValueError("`resolution` must be divisible by 8 for SDXL latent scaling.")
    if not 0.0 <= args.background_fill_value <= 1.0:
        raise ValueError("`background_fill_value` must be in [0, 1].")
    if args.camera_azimuth_bins < 1:
        raise ValueError("`camera_azimuth_bins` must be at least 1.")
    if args.camera_azimuth_jitter < 0.0:
        raise ValueError("`camera_azimuth_jitter` must be non-negative.")
    validate_triangular_range(
        "camera_elevation",
        args.camera_elevation_min,
        args.camera_elevation_max,
        args.camera_elevation_mode,
    )
    validate_triangular_range(
        "camera_distance",
        args.camera_distance_min,
        args.camera_distance_max,
        args.camera_distance_mode,
    )
    validate_triangular_range(
        "light_elevation",
        args.light_elevation_min,
        args.light_elevation_max,
        args.light_elevation_mode,
    )
    validate_triangular_range(
        "light_distance",
        args.light_distance_min,
        args.light_distance_max,
        args.light_distance_mode,
    )
    if args.light_azimuth_max <= args.light_azimuth_min:
        raise ValueError(
            "`light_azimuth_max` must be greater than `light_azimuth_min`."
        )
    return args


def ensure_required_paths(args):
    required_paths = {
        "mesh_obj_path": args.mesh_obj_path,
        "uv_mask_path": args.uv_mask_path,
        "reward_model_path": args.reward_model_path,
    }
    for label, path in required_paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"`{label}` does not exist: {path}")


def validate_triangular_range(name: str, low: float, high: float, mode: float) -> None:
    if high < low:
        raise ValueError(f"`{name}` has invalid bounds: {low} > {high}.")
    if not low <= mode <= high:
        raise ValueError(
            f"`{name}_mode` must be within [`{name}_min`, `{name}_max`], got {mode}."
        )


def sample_triangular(low: float, high: float, mode: float) -> float:
    if high == low:
        return low
    return random.triangular(low, high, mode)


class RenderSamplingState:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self._azimuth_order: List[int] = []
        self._azimuth_index = 0

    def _next_azimuth_bin(self) -> int:
        if self._azimuth_index >= len(self._azimuth_order):
            self._azimuth_order = list(range(self.args.camera_azimuth_bins))
            random.shuffle(self._azimuth_order)
            self._azimuth_index = 0
        bin_index = self._azimuth_order[self._azimuth_index]
        self._azimuth_index += 1
        return bin_index

    def sample_camera(self):
        if self.args.camera_sampler == "legacy":
            elev = random.choice([0.0, 10.0, 20.0])
            azim = random.choice(list(range(0, 360, 10)))
            # dist = random.uniform(7.0, 12.0)
            dist = random.choice([7.0, 8.0, 9.0, 10.0])
            return dist, elev, azim
        # raise NotImplementedError(f"Unsupported camera sampler: {self.args.camera_sampler}")
        bin_index = self._next_azimuth_bin()
        bin_width = 360.0 / self.args.camera_azimuth_bins
        bin_center = (bin_index + 0.5) * bin_width
        max_jitter = bin_width / 2.0
        jitter = min(self.args.camera_azimuth_jitter, max_jitter)
        azim = (bin_center + random.uniform(-jitter, jitter)) % 360.0
        elev = sample_triangular(
            self.args.camera_elevation_min,
            self.args.camera_elevation_max,
            self.args.camera_elevation_mode,
        )
        dist = sample_triangular(
            self.args.camera_distance_min,
            self.args.camera_distance_max,
            self.args.camera_distance_mode,
        )
        return dist, elev, azim

    def sample_light_location(self) -> torch.Tensor:
        dist = sample_triangular(
            self.args.light_distance_min,
            self.args.light_distance_max,
            self.args.light_distance_mode,
        )
        elev = sample_triangular(
            self.args.light_elevation_min,
            self.args.light_elevation_max,
            self.args.light_elevation_mode,
        )
        azim = random.uniform(self.args.light_azimuth_min, self.args.light_azimuth_max)
        return camera_position_from_spherical_angles(
            distance=dist,
            elevation=elev,
            azimuth=azim,
            degrees=True,
            device=self.device,
        )


def render_reward_multi_view(
    texture_render_model,
    render_sampling_state: RenderSamplingState,
    yolo_reward_model,
    diffusion_images: torch.Tensor,
    background,
    num_views: int,
    randomize_lighting: bool,
    reward_mode: str = "non_targeted",
    target_class_id: int = 2,
):
    rendered_views = []
    reward_values = []
    reward_statuses = []
    view_cameras = []

    for _ in range(num_views):
        dist, elev, azim = render_sampling_state.sample_camera()
        view_cameras.append((dist, elev, azim))
        cam = texture_render_model.get_camera(dist=dist, elev=elev, azim=azim)
        if randomize_lighting:
            texture_render_model.set_light_location(
                render_sampling_state.sample_light_location()
            )
        rendered = texture_render_model(cam, diffusion_images, background)
        rendered = rendered.permute(0, 3, 1, 2).clamp(0, 1)
        reward_status, reward_value = yolo_reward_model(
            rendered, reward_mode, target_class_id
        )
        rendered_views.append(rendered)
        reward_values.append(reward_value.reshape(()))
        reward_statuses.append(reward_status)

    reward_tensor = torch.stack(reward_values)
    mean_reward = reward_tensor.mean()
    any_detected = any(reward_statuses)

    return {
        "reward_status": any_detected,
        "reward_value": mean_reward,
        "reward_values": reward_tensor,
        "rendered_views": rendered_views,
        "view_cameras": view_cameras,
    }


def resolve_unet_lora_target_modules(unet, lora_scope: str) -> List[str]:
    if lora_scope not in {"attn1", "attn2", "both"}:
        raise ValueError(f"Unsupported LoRA scope: {lora_scope}")

    target_modules = []
    for module_name, _ in unet.named_modules():
        if not module_name or not module_name.endswith(LORA_TARGET_LEAF_MODULES):
            continue
        if lora_scope == "attn1" and ".attn1." not in module_name:
            continue
        if lora_scope == "attn2" and ".attn2." not in module_name:
            continue
        target_modules.append(module_name)

    target_modules = sorted(set(target_modules))
    if not target_modules:
        raise ValueError(
            f"No UNet modules matched LoRA scope `{lora_scope}` with leaves {LORA_TARGET_LEAF_MODULES}."
        )
    return target_modules


def build_optimizer(args, params: Iterable[torch.nn.Parameter]):
    return torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


def unwrap_model(accelerator: Accelerator, model):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def prepare_scheduler_ddim(base_scheduler, num_inference_steps: int):
    scheduler = DDIMScheduler.from_config(base_scheduler.config)
    scheduler.set_timesteps(num_inference_steps)
    return scheduler


def prepare_scheduler_dpm(base_scheduler, num_inference_steps: int):
    scheduler = DPMSolverMultistepScheduler()
    scheduler.set_timesteps(num_inference_steps)
    return scheduler


def make_text_embeddings(
    tokenizer_one,
    tokenizer_two,
    text_encoder_one,
    text_encoder_two,
    prompts: Sequence[str],
    negative_prompt: str,
    tokenizer_max_length: Optional[int],
    use_attention_mask: bool,
):
    text_inputs_one = tokenize_prompt(
        tokenizer_one, list(prompts), tokenizer_max_length
    )
    prompt_embeds_one, _ = encode_prompt_with_text_encoder(
        text_encoder_one,
        text_inputs_one.input_ids,
        text_inputs_one.attention_mask,
        text_encoder_use_attention_mask=use_attention_mask,
    )

    text_inputs_two = tokenize_prompt(
        tokenizer_two, list(prompts), tokenizer_max_length
    )
    prompt_embeds_two, pooled_prompt_embeds = encode_prompt_with_text_encoder(
        text_encoder_two,
        text_inputs_two.input_ids,
        text_inputs_two.attention_mask,
        text_encoder_use_attention_mask=use_attention_mask,
    )

    prompt_embeds = torch.concat([prompt_embeds_one, prompt_embeds_two], dim=-1)

    negative_prompts = [negative_prompt] * len(prompts)
    negative_inputs_one = tokenize_prompt(
        tokenizer_one, negative_prompts, tokenizer_max_length
    )
    negative_embeds_one, _ = encode_prompt_with_text_encoder(
        text_encoder_one,
        negative_inputs_one.input_ids,
        negative_inputs_one.attention_mask,
        text_encoder_use_attention_mask=use_attention_mask,
    )

    negative_inputs_two = tokenize_prompt(
        tokenizer_two, negative_prompts, tokenizer_max_length
    )
    negative_embeds_two, negative_pooled_prompt_embeds = (
        encode_prompt_with_text_encoder(
            text_encoder_two,
            negative_inputs_two.input_ids,
            negative_inputs_two.attention_mask,
            text_encoder_use_attention_mask=use_attention_mask,
        )
    )

    negative_prompt_embeds = torch.concat(
        [negative_embeds_one, negative_embeds_two], dim=-1
    )
    return (
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    )


def build_sdxl_time_ids(
    batch_size: int, resolution: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    # SDXL micro-conditioning: [original_h, original_w, crop_y, crop_x, target_h, target_w]
    add_time_ids = torch.tensor(
        [resolution, resolution, 0, 0, resolution, resolution],
        device=device,
        dtype=dtype,
    )
    return add_time_ids.unsqueeze(0).repeat(batch_size, 1)


def sample_latents_with_grad(
    unet,
    vae,
    scheduler,
    prompt_embeds,
    pooled_prompt_embeds,
    negative_prompt_embeds,
    negative_pooled_prompt_embeds,
    add_time_ids,
    device,
    dtype,
    batch_size: int,
    resolution: int,
    guidance_scale: float,
    truncated_backprop_rand: bool = True,
    truncated_backprop_minmax: tuple = (0, 50),
):
    """
    执行 DDIM 采样，并在随机截断点 K 之后保留梯度计算图。

    - timestep > K（早期去噪步）：stop_gradient，不参与反向传播
    - timestep <= K（后期去噪步）：保留计算图，梯度可以流过

    注意：scheduler.timesteps 是降序的，例如 [999, 979, ..., 19, 0]
    所以 "早期步" 对应大 timestep，"后期步" 对应小 timestep。
    K 控制从第几个 step 开始保留梯度（从后往前数）。
    """
    latent_size = resolution // 8
    target_dtype = dtype

    # 采样初始噪声 x_T
    latents = torch.randn(
        (batch_size, 4, latent_size, latent_size),
        device=device,
        dtype=target_dtype,
    )
    latents = (latents * scheduler.init_noise_sigma).to(dtype=target_dtype)

    # 随机采样截断点 K：从第 K 步开始才保留梯度
    total_steps = len(scheduler.timesteps)
    if truncated_backprop_rand:
        K = random.randint(
            truncated_backprop_minmax[0],
            min(truncated_backprop_minmax[1], total_steps - 1),
        )
    else:
        K = truncated_backprop_minmax[1]  # 固定截断点
    K = 18
    for step_idx, timestep in enumerate(scheduler.timesteps):
        # step_idx < K：截断，不保留梯度（减少显存，防止 mode collapse）
        # step_idx >= K：保留梯度，参与 reward backprop
        if step_idx < K:
            with torch.no_grad():
                latents = _ddim_step(
                    unet,
                    scheduler,
                    latents,
                    timestep,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    add_time_ids,
                    guidance_scale,
                    target_dtype,
                )
                latents = latents.detach()  # 显式断开计算图
        else:
            # gradient checkpointing 在外部通过 unet.enable_gradient_checkpointing() 控制
            latents = _ddim_step(
                unet,
                scheduler,
                latents,
                timestep,
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                add_time_ids,
                guidance_scale,
                target_dtype,
            )

    # VAE 解码：latent → pixel image [0, 1]
    # VAE 解码不需要梯度（reward 直接作用在 pixel 上）
    # SDXL 的 VAE 在半精度下容易出现数值问题（黑图/NaN），解码时与 VAE dtype 对齐。
    vae_decode_dtype = vae.dtype if hasattr(vae, "dtype") else torch.float32
    images = vae.decode(
        (latents / vae.config.scaling_factor).to(dtype=vae_decode_dtype),
        return_dict=False,
    )[0]
    images = (images / 2 + 0.5).clamp(0, 1)

    return images


def _ddim_step(
    unet,
    scheduler,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    negative_pooled_prompt_embeds: torch.Tensor,
    add_time_ids: torch.Tensor,
    guidance_scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """单步 DDIM 去噪（含 classifier-free guidance）"""
    latent_model_input = torch.cat([latents, latents], dim=0)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep).to(
        dtype=dtype
    )

    encoder_hidden_states = torch.cat(
        [negative_prompt_embeds, prompt_embeds], dim=0
    ).to(dtype=dtype)
    text_embeds = torch.cat(
        [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
    ).to(dtype=dtype)
    time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(
        device=latents.device, dtype=dtype
    )

    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
        return_dict=False,
    )[0]

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )
    guided_noise_pred = guided_noise_pred.to(dtype=dtype)

    latents = scheduler.step(guided_noise_pred, timestep, latents, return_dict=False)[0]
    return latents.to(dtype=dtype)


def prune_checkpoints(output_dir: str, total_limit: Optional[int]):
    if total_limit is None:
        return
    checkpoints = sorted(
        [path for path in os.listdir(output_dir) if path.startswith("checkpoint-")],
        key=lambda item: int(item.split("-")[1]),
    )
    if len(checkpoints) < total_limit:
        return
    remove_count = len(checkpoints) - total_limit + 1
    for checkpoint in checkpoints[:remove_count]:
        shutil.rmtree(Path(output_dir) / checkpoint, ignore_errors=True)


def iter_prompt_groups(prompt_groups):
    for camo_type, sections in prompt_groups.items():
        for section_name, prompts in sections.items():
            for prompt_index, prompt in enumerate(prompts):
                yield camo_type, section_name, prompt_index, prompt


def parse_prompt_specs(prompt_specs: Sequence[str], available_keys: Sequence[str]):
    if not prompt_specs:
        prompt_specs = ["all:1.0"]

    # 支持空格分隔和逗号分隔混用，例如: --prompt bio:0.2,arc:0.3
    tokens: List[str] = []
    for raw in prompt_specs:
        for part in raw.split(","):
            token = part.strip()
            if token:
                tokens.append(token)

    if not tokens:
        raise ValueError("`--prompt` is empty after parsing.")

    available_set = set(available_keys)
    ratio_by_key = {key: 0.0 for key in available_keys}

    for token in tokens:
        if ":" not in token:
            raise ValueError(
                f"Invalid prompt spec `{token}`. Expected format `name:ratio`."
            )
        name, ratio_text = token.split(":", 1)
        name = name.strip().lower()
        ratio_text = ratio_text.strip()
        try:
            ratio = float(ratio_text)
        except ValueError as exc:
            raise ValueError(f"Invalid ratio in prompt spec `{token}`.") from exc
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(
                f"Prompt ratio must be in [0, 1], got {ratio} from `{token}`."
            )

        if name == "all":
            for key in ratio_by_key:
                ratio_by_key[key] = ratio
            continue

        if name not in available_set:
            raise ValueError(
                f"Unknown prompt group `{name}`. Available: {sorted(available_set)} and `all`."
            )
        ratio_by_key[name] = ratio

    return ratio_by_key


def build_prompt_list_from_specs(all_prompts, prompt_specs: Sequence[str]):
    keys = list(all_prompts.keys())
    ratio_by_key = parse_prompt_specs(prompt_specs, keys)

    selected_prompts: List[str] = []
    selected_counts = {}
    for key in keys:
        prompts = all_prompts[key]
        ratio = ratio_by_key.get(key, 0.0)
        take_count = int(len(prompts) * ratio)
        if ratio >= 1.0:
            take_count = len(prompts)
        selected = prompts[:take_count]
        selected_prompts.extend(selected)
        selected_counts[key] = {
            "ratio": ratio,
            "take": take_count,
            "total": len(prompts),
        }

    if not selected_prompts:
        raise ValueError(
            "No prompts selected by `--prompt`. Increase ratio(s) or use `all:1.0`."
        )

    return selected_prompts, selected_counts


@torch.inference_mode()
def run_quick_asr_eval(
    global_step: int,
    checkpoint_dir: Path,
    args,
    accelerator,
    tokenizer_one,
    tokenizer_two,
    text_encoder_one,
    text_encoder_two,
    vae,
    unet,
    base_scheduler,
    weight_dtype,
    texture_render_model,
    yolo_model,
):
    unet.eval()
    quick_cam = {
        "dist": [7.0, 8.0, 9.0, 10.0],
        "elev": [0.0, 10.0, 20.0],
        "azim": list(range(0, 360, 30)),
    }
    backgrounds_quick = [
        "data/background/desert/desert_00011.jpg",
        "data/background/jungle/jungle_00020.jpg",
        "data/background/sunset/sunset_00004.jpg",
        "data/background/snow/snow_00002.jpg",
    ]
    quick_eval_dir = checkpoint_dir / "quick_eval_images"
    quick_eval_dir.mkdir(parents=True, exist_ok=True)
    quick_eval_log_path = checkpoint_dir / f"quick_eval_{global_step}.log"

    background_images = {
        background_path: Image.open(background_path).convert("RGB")
        for background_path in backgrounds_quick
    }
    prompt_groups = get_prompts_by_speed("quick", clean=True)
    texture_render_model.set_light_location(DEFAULT_LIGHT_LOCATION)

    with quick_eval_log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("global_step#type#prompt#total_views#detection_rate\n")
        ALL_detections = 0
        ALL_views = 0
        for camo_type, section_name, prompt_index, prompt in iter_prompt_groups(
            prompt_groups
        ):
            logger.info(
                "Quick ASR eval: type=%s/%s, prompt=%s",
                camo_type,
                section_name,
                prompt,
            )
            scheduler = prepare_scheduler_ddim(base_scheduler, args.num_inference_steps)
            # scheduler = base_scheduler
            # scheduler.set_timesteps(args.num_inference_steps)
            (
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = make_text_embeddings(
                tokenizer_one,
                tokenizer_two,
                text_encoder_one,
                text_encoder_two,
                [prompt],
                args.negative_prompt,
                args.tokenizer_max_length,
                args.text_encoder_use_attention_mask,
            )
            prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
            pooled_prompt_embeds = pooled_prompt_embeds.to(
                accelerator.device, dtype=weight_dtype
            )
            negative_prompt_embeds = negative_prompt_embeds.to(
                accelerator.device, dtype=weight_dtype
            )
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(
                accelerator.device, dtype=weight_dtype
            )
            add_time_ids = build_sdxl_time_ids(
                batch_size=1,
                resolution=args.resolution,
                device=accelerator.device,
                dtype=weight_dtype,
            )

            with torch.no_grad():
                diffusion_image = sample_latents_with_grad(
                    unet,
                    vae,
                    scheduler,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    add_time_ids,
                    accelerator.device,
                    weight_dtype,
                    1,
                    args.resolution,
                    args.guidance_scale,
                )

            prompt_tag = f"{camo_type}_{section_name}_{prompt_index:02d}"
            texture_image_path = quick_eval_dir / f"{prompt_tag}.png"
            prompt_text_path = quick_eval_dir / f"{prompt_tag}.txt"
            tensor_rgb_to_pil(diffusion_image).save(texture_image_path)
            prompt_text_path.write_text(prompt, encoding="utf-8")

            for background_path, background_image in background_images.items():
                type_key = f"{camo_type}/{section_name}/{Path(background_path).stem}"
                asr_dict = TestASRQuick(
                    diffusion_image.detach(),
                    background=background_image,
                    cam_params=quick_cam,
                    model=yolo_model.model,
                    texture_render_model=texture_render_model,
                    device=str(accelerator.device),
                )
                detection_rate = asr_dict["detection_rate"]
                total_views = asr_dict["total_views"]
                detected = asr_dict["detected"]
                not_detected = asr_dict["not_detected"]
                ALL_detections += detected
                ALL_views += total_views
                log_line = f"{global_step}#{type_key}#{prompt}#{total_views}#{detection_rate}\n"
                log_file.write(log_line)
                log_file.flush()
                logger.info(
                    "Quick ASR result: type=%s, total_views=%s, detection_rate=%s",
                    type_key,
                    total_views,
                    detection_rate,
                )
        ALL_rate = ALL_detections / ALL_views
        log_line = f"{global_step}#{ALL_detections}#{ALL_views}#{ALL_rate:.4f}\n"
        log_file.write(log_line)
        log_file.flush()


def run_quick_asr_eval_pipeline(
    global_step: int,
    checkpoint_dir: Path,
    args,
    accelerator,
    tokenizer_one,
    tokenizer_two,
    text_encoder_one,
    text_encoder_two,
    vae,
    unet,
    base_scheduler,
    weight_dtype,
    texture_render_model,
    yolo_model,
):
    unet.eval()
    quick_cam = {
        "dist": [7.0, 8.0, 9.0, 10.0],
        "elev": [0.0, 10.0, 20.0],
        "azim": list(range(0, 360, 30)),
    }
    backgrounds_quick = [
        "data/background/desert/desert_00011.jpg",
        "data/background/jungle/jungle_00020.jpg",
        "data/background/sunset/sunset_00004.jpg",
        "data/background/snow/snow_00002.jpg",
    ]
    quick_eval_dir = checkpoint_dir / "quick_eval_images"
    quick_eval_dir.mkdir(parents=True, exist_ok=True)
    quick_eval_log_path = checkpoint_dir / f"quick_eval_{global_step}.log"

    background_images = {
        background_path: Image.open(background_path).convert("RGB")
        for background_path in backgrounds_quick
    }
    prompt_groups = get_prompts_by_speed("quick", clean=True)
    texture_render_model.set_light_location(DEFAULT_LIGHT_LOCATION)

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        vae=vae,
        unet=unet,
        scheduler=prepare_scheduler_ddim(base_scheduler, args.num_inference_steps),
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    try:
        with quick_eval_log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("global_step#type#prompt#total_views#detection_rate\n")
            ALL_detections = 0
            ALL_views = 0
            for camo_type, section_name, prompt_index, prompt in iter_prompt_groups(
                prompt_groups
            ):
                logger.info(
                    "Quick ASR eval (pipeline): type=%s/%s, prompt=%s",
                    camo_type,
                    section_name,
                    prompt,
                )

                with torch.no_grad():
                    diffusion_image = pipeline(
                        prompt=prompt,
                        negative_prompt=args.negative_prompt,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        height=args.resolution,
                        width=args.resolution,
                    ).images[0]  # type: ignore

                prompt_tag = f"{camo_type}_{section_name}_{prompt_index:02d}"
                texture_image_path = quick_eval_dir / f"{prompt_tag}.png"
                prompt_text_path = quick_eval_dir / f"{prompt_tag}.txt"
                diffusion_image.save(texture_image_path)
                prompt_text_path.write_text(prompt, encoding="utf-8")

                for background_path, background_image in background_images.items():
                    type_key = (
                        f"{camo_type}/{section_name}/{Path(background_path).stem}"
                    )
                    asr_dict = TestASRQuick(
                        diffusion_image,
                        background=background_image,
                        cam_params=quick_cam,
                        model=yolo_model.model,
                        texture_render_model=texture_render_model,
                        device=str(accelerator.device),
                    )
                    detection_rate = asr_dict["detection_rate"]
                    total_views = asr_dict["total_views"]
                    detected = asr_dict["detected"]
                    not_detected = asr_dict["not_detected"]
                    ALL_detections += detected
                    ALL_views += total_views
                    log_line = f"{global_step}#{type_key}#{prompt}#{total_views}#{detection_rate}\n"
                    log_file.write(log_line)
                    log_file.flush()
                    logger.info(
                        "Quick ASR result (pipeline): type=%s, total_views=%s, detection_rate=%s",
                        type_key,
                        total_views,
                        detection_rate,
                    )
            ALL_rate = ALL_detections / ALL_views
            log_line = f"{global_step}#{ALL_detections}#{ALL_views}#{ALL_rate:.4f}\n"
            log_file.write(log_line)
            log_file.flush()
    finally:
        del pipeline
        free_memory()


def main(args):
    ensure_required_paths(args)

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb" and not is_wandb_available():
        raise ImportError("Install wandb to use `--report_to wandb`.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    tracker_name = f"fastadv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    temp_args = vars(args).copy()
    temp_args["prompt"] = str(temp_args["prompt"])
    accelerator.init_trackers(tracker_name, config=temp_args)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name
        if args.tokenizer_name
        else args.pretrained_model_name_or_path,
        subfolder=None if args.tokenizer_name else "tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )
    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path,
        args.revision,
        subfolder="text_encoder",
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path,
        args.revision,
        subfolder="text_encoder_2",
    )
    base_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )
    yolo_reward_model = YoloRewardModel(
        args.reward_model_path,
    )
    # 加载clip模型
    clip_model = CLIPModel.from_pretrained(
        "/home/dk/data/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/"
    )
    clip_processor = CLIPProcessor.from_pretrained(
        "/home/dk/data/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/"
    )
    clip_model.eval()

    text_encoder.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    clip_model.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # SDXL 训练中保持 VAE 为 fp32，避免解码出现黑图/NaN。
    vae_dtype = torch.float32

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=vae_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    yolo_reward_model.to(accelerator.device)
    clip_model.to(accelerator.device, dtype=weight_dtype)

    uv_mask = load_uv_mask(
        args.uv_mask_path, args.texture_image_size, accelerator.device
    )

    texture_render_model = TextureRenderModel(
        device=accelerator.device,
        mesh_obj_path=args.mesh_obj_path,
        uv_mask=uv_mask,
        render_image_size=args.render_image_size,
        texture_image_size=args.texture_image_size,
        background_fill_value=args.background_fill_value,
    )

    texture_render_model.to(accelerator.device)
    render_sampling_state = RenderSamplingState(args, accelerator.device)
    logger.info(
        "Camera sampler `%s` configured with %d reward views, elevation=[%.1f, %.1f] mode=%.1f, distance=[%.1f, %.1f] mode=%.1f, azimuth_bins=%d.",
        args.camera_sampler,
        args.num_reward_views,
        args.camera_elevation_min,
        args.camera_elevation_max,
        args.camera_elevation_mode,
        args.camera_distance_min,
        args.camera_distance_max,
        args.camera_distance_mode,
        args.camera_azimuth_bins,
    )
    if args.randomize_lighting:
        logger.info(
            "Lighting randomization enabled with elevation=[%.1f, %.1f] mode=%.1f, distance=[%.1f, %.1f] mode=%.1f, azimuth=[%.1f, %.1f].",
            args.light_elevation_min,
            args.light_elevation_max,
            args.light_elevation_mode,
            args.light_distance_min,
            args.light_distance_max,
            args.light_distance_mode,
            args.light_azimuth_min,
            args.light_azimuth_max,
        )
    else:
        logger.info(
            "Lighting randomization disabled; using fixed light at %s.",
            DEFAULT_LIGHT_LOCATION,
        )

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    lora_target_modules = resolve_unet_lora_target_modules(unet, args.lora_scope)
    logger.info(
        "LoRA scope `%s` matched %d UNet modules.",
        args.lora_scope,
        len(lora_target_modules),
    )
    if args.lora_scope == "both":
        print(
            "重要:LoRA will be applied to both attn1 and attn2 modules, which may increase training time and memory usage but can potentially improve reward optimization."
        )
        lora_target_modules = [
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "add_k_proj",
            "add_v_proj",
        ]
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        # target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
        target_modules=lora_target_modules,
    )
    unet.add_adapter(unet_lora_config)
    cast_training_params([unet], dtype=torch.float32)

    trainable_lora_names = [
        name
        for name, parameter in unet.named_parameters()
        if parameter.requires_grad and "lora" in name.lower()
    ]
    logger.info(
        "Enabled %d trainable LoRA tensors for scope `%s`.",
        len(trainable_lora_names),
        args.lora_scope,
    )
    # for name in trainable_lora_names:
    #     logger.info("LoRA trainable param: %s", name)

    params_to_optimize = [
        parameter for parameter in unet.parameters() if parameter.requires_grad
    ]
    optimizer = build_optimizer(args, params_to_optimize)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    # 加载背景数据集
    dataloader = get_background_dataloader(
        # "data/background", category=["jungle", "sunset","snow"], batch_size=1
        "data/background",
        # category=["sunset"],
        category=["jungle", "sunset", "snow", "desert"],
        batch_size=1,
    )
    natural_texture_colors = [
        # 绿色系（植被/爬虫/丛林）
        "olive green",
        "forest green",
        "moss green",
        "sage green",
        "pine green",
        "emerald green",
        "jade green",
        "fern green",
        # 棕褐色系（大地/皮毛/树皮）
        "taupe brown",
        "chestnut brown",
        "walnut brown",
        "mahogany brown",
        "caramel brown",
        "umber brown",
        "sepia brown",
        # 黄/土色系（沙漠/沙土/斑纹）
        "sand beige",
        "khaki beige",
        "ochre yellow",
        "amber yellow",
        "wheat yellow",
        "mustard yellow",
        # 灰/黑/白色系（雪地/岩石/斑纹）
        "slate grey",
        "ash grey",
        "silver grey",
        "charcoal black",
        "ivory white",
        "snow white",
        "cream off-white",
        # 蓝/青/冷色系（水生/昆虫/深海）
        "teal blue",
        "slate blue",
        "navy blue",
        "cerulean blue",
        # 红/橙/金属色系（斑纹/甲壳/鳞片）
        "rust red",
        "terracotta red",
        "burnt orange",
        "copper orange",
    ]
    unet, optimizer, lr_scheduler, dataloader = accelerator.prepare(
        unet, optimizer, lr_scheduler, dataloader
    )

    global_step = 0
    from data.prompt import ALL_PROMPTS

    total_prompts, selected_counts = build_prompt_list_from_specs(
        ALL_PROMPTS, args.prompt
    )
    logger.info("Prompt selection specs: %s", args.prompt)
    for prompt_key, stats in selected_counts.items():
        logger.info(
            "Prompt group `%s`: ratio=%.3f, selected=%d/%d",
            prompt_key,
            stats["ratio"],
            stats["take"],
            stats["total"],
        )
    logger.info("Total selected prompts: %d", len(total_prompts))
    # for step in range(args.max_train_steps):
    # total_prompts = mix 
    stop_training = False
    for epoch in range(1, 100000):
        for idx, bgdata in enumerate(dataloader):
            scheduler = prepare_scheduler_ddim(base_scheduler, args.num_inference_steps)
            # scheduler = base_scheduler
            # scheduler.set_timesteps(args.num_inference_steps)
            # scheduler.set_timesteps(random.randint(15,35))
            color = f", {random.choice(natural_texture_colors)}"
            prompts = [random.choice(total_prompts) + color]
            with accelerator.accumulate(unet):
                (
                    prompt_embeds,
                    pooled_prompt_embeds,
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = make_text_embeddings(
                    tokenizer,
                    tokenizer_two,
                    text_encoder,
                    text_encoder_two,
                    prompts,
                    args.negative_prompt,
                    args.tokenizer_max_length,
                    args.text_encoder_use_attention_mask,
                )
                prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(
                    accelerator.device, dtype=weight_dtype
                )
                negative_prompt_embeds = negative_prompt_embeds.to(
                    accelerator.device, dtype=weight_dtype
                )
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(
                    accelerator.device, dtype=weight_dtype
                )
                add_time_ids = build_sdxl_time_ids(
                    batch_size=args.reward_batch_size,
                    resolution=args.resolution,
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                # bchw
                diffusion_images = sample_latents_with_grad(
                    unet,
                    vae,
                    scheduler,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    add_time_ids,
                    accelerator.device,
                    weight_dtype,
                    args.reward_batch_size,
                    args.resolution,
                    args.guidance_scale,
                )

                reward_outputs = render_reward_multi_view(
                    texture_render_model=texture_render_model,
                    render_sampling_state=render_sampling_state,
                    yolo_reward_model=yolo_reward_model,
                    diffusion_images=diffusion_images,
                    background=bgdata,
                    num_views=args.num_reward_views,
                    randomize_lighting=args.randomize_lighting,
                    reward_mode="non_targeted",
                    target_class_id=2,
                )

                # dist, elev, azim = render_sampling_state.sample_camera()
                # cam = texture_render_model.get_camera(dist=dist, elev=elev, azim=azim)
                # rendered = texture_render_model(cam, diffusion_images, bgdata)
                # # rendered = texture_render_model(cam, diffusion_images)
                # rendered = rendered.permute(0, 3, 1, 2).clamp(0, 1)
                # rewardStatus, reward_value = yolo_reward_model(
                #     rendered, "non_targeted", 2
                # )

                rendered = reward_outputs["rendered_views"][0]
                rewardStatus = reward_outputs["reward_status"]
                reward_value = reward_outputs["reward_value"]
                reward_values = reward_outputs["reward_values"]

                # 每20步保存一次渲染结果
                # if accelerator.is_main_process and global_step % 20 == 0:
                if accelerator.is_main_process and idx % 20 == 0:
                    texture_min = diffusion_images.detach().amin().item()
                    texture_max = diffusion_images.detach().amax().item()
                    texture_mean = diffusion_images.detach().mean().item()
                    logger.info(
                        "Step %s: diffusion_images stats min=%.6f max=%.6f mean=%.6f",
                        global_step,
                        texture_min,
                        texture_max,
                        texture_mean,
                    )
                    rendered_image = tensor_rgb_to_pil(rendered)
                    texture_image = tensor_rgb_to_pil(diffusion_images)

                    rendered_image.save(
                        f"{args.output_dir}/{epoch}_rendered_images_{idx}.png"
                    )
                    texture_image.save(
                        f"{args.output_dir}/{epoch}_texture_images_{idx}.png"
                    )

                    combined_image = Image.new(
                        "RGB",
                        (
                            rendered_image.width + texture_image.width,
                            max(rendered_image.height, texture_image.height),
                        ),
                    )
                    combined_image.paste(rendered_image, (0, 0))
                    combined_image.paste(texture_image, (rendered_image.width, 0))
                    combined_image.save(
                        f"{args.output_dir}/{epoch}_combined_images_{idx}.png"
                    )
                clip_score = clip_text_image_similarity(
                    clip_processor,
                    clip_model,
                    diffusion_images,
                    prompts,
                    accelerator.device,
                    needgrad=True,
                )
                # clip正则化项
                loss_clip_score = args.clip_weight * (1 - clip_score)  # -1,1 -> 0,2,
                loss_reg = args.lora_reg_weight * lora_regularization(unet)
                loss_reward = args.reward_weight * reward_value
                loss_reward_active = (
                    loss_reward if rewardStatus else torch.zeros_like(loss_reward)
                )
                if not rewardStatus:
                    logger.info(
                        f"Step {global_step}: 攻击成功, CLIP Score:   {clip_score.item():.4f}"
                    )
                    loss = loss_reg + (1 - clip_score)
                else:
                    loss = loss_reward_active + loss_reg + loss_clip_score

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                # optimizer.zero_grad(set_to_none=False)
                optimizer.zero_grad()
                if accelerator.is_main_process and global_step % 1 == 0:
                    reward_values_text = ", ".join(
                        f"{value:.4f}"
                        for value in reward_values.detach().cpu().tolist()
                    )
                    # reward_values_text = f"{reward_value.item():.4f}"
                    logger.info(
                        f"Step {global_step}: Loss={loss.detach().item():.4f}, LossReward={loss_reward_active.detach().item():.4f}, LossReg={loss_reg.detach().item():.4f}, LossClip={loss_clip_score.detach().item():.4f}, RewardMean={reward_value.detach().item():.4f}, RewardViews=[{reward_values_text}], LR={lr_scheduler.get_last_lr()[0]:.6f}"
                    )
                    logger.info(
                        f"Step {global_step}: Prompts: {prompts}, CLIP Score:   {clip_score.item():.4f}"
                    )

            if accelerator.sync_gradients:
                global_step += 1
                logs = {
                    "loss": loss.detach().item(),
                    "loss_reward": loss_reward_active.detach().item(),
                    "loss_reg": loss_reg.detach().item(),
                    "loss_clip_score": loss_clip_score.detach().item(),
                    "reward": reward_value.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                accelerator.log(logs, step=global_step)
                if global_step >= args.max_train_steps:
                    stop_training = True
                    break
                # 每 N次 梯度步保存一次 LoRA 权重检查点，并进行评估
                if (
                    accelerator.is_main_process
                    and global_step % args.checkpointing_steps == 0
                ):
                    prune_checkpoints(args.output_dir, args.checkpoints_total_limit)
                    checkpoint_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    unwrapped_unet = unwrap_model(accelerator, unet)
                    # unwrapped_unet = unwrap_model(accelerator, unet).to(torch.float32)
                    unet_lora_state_dict = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(unwrapped_unet)
                    )
                    StableDiffusionXLPipeline.save_lora_weights(
                        save_directory=checkpoint_dir,
                        unet_lora_layers=unet_lora_state_dict,
                    )
                    torch.cuda.empty_cache()

                    # run_quick_asr_eval(
                    #     global_step=global_step,
                    #     checkpoint_dir=checkpoint_dir,
                    #     args=args,
                    #     accelerator=accelerator,
                    #     tokenizer=tokenizer,
                    #     text_encoder=text_encoder,
                    #     vae=vae,
                    #     unet=unwrapped_unet,
                    #     base_scheduler=base_scheduler,
                    #     weight_dtype=weight_dtype,
                    #     texture_render_model=texture_render_model,
                    #     yolo_model=yolo_reward_model,
                    # )
        if stop_training:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_unet = unwrap_model(accelerator, unet).to(torch.float32)
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(final_unet)
        )
        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
        )
        free_memory()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
