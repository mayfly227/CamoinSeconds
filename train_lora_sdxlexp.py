import os
from datetime import datetime

from data.dataset import build_cameras
from prompt_pool import PromptPool
from prompt_poolcodex import PromptPool as PromptPoolCodex
from utils.aug import ROA, ROADiffusionImage
from utils.core import get_free_gpu, load_uv_mask
from utils.loss import (
    color_distribution_loss,
    disable_lora_context,
    diversity_repulsion_loss,
    lora_regularization,
    margin_detection_loss,
    soft_color_histogram,
    trajectory_anchor_loss,
)

try:
    get_free_gpu()
except Exception as e:  # noqa: BLE001
    print(
        f"Warning: get_free_gpu() failed with error: {e}. Proceeding without setting CUDA_VISIBLE_DEVICES."
    )
    exit(0)

import argparse
import logging
import random
import shutil
from collections import deque
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional  # noqa: UP035

import diffusers
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
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
)
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor, PretrainedConfig

from meshrender import TextureRenderModel, tensor_rgb_to_pil
from utils.cliputil import clip_image_feature, clip_text_image_similarity

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
        default="/mnt/data/dk/hf/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b",
    )
    parser.add_argument(
        "--clip_path",
        type=str,
        default="/mnt/data/dk/hf/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41",
    )
    # 模型版本号或分支名
    parser.add_argument("--revision", type=str, default=None)
    # 模型权重变体，例如 fp16
    parser.add_argument("--variant", type=str, default=None)
    # 单独指定 tokenizer 路径
    parser.add_argument("--tokenizer_name", type=str, default=None)
    # 训练输出目录
    parser.add_argument("--output_dir", type=str, default="rewardyolosdxl")
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
    parser.add_argument("--use_margin_loss", action="store_true")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.1,
        help="Detection confidence margin used by --use_margin_loss.",
    )
    # 梯度累积步数
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # 是否启用梯度检查点节省显存
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_old_data", action="store_true")
    parser.add_argument("--use_old_anchor", action="store_true")
    # 是否启用 xFormers 高效注意力
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", type=bool, default=False
    )
    parser.add_argument(
        "--nvdiffrast_texture_filter",
        type=str,
        default="linear-mipmap-linear",
        choices=["nearest", "linear", "linear-mipmap-nearest", "linear-mipmap-linear"],
        help=(
            "nvdiffrast texture filtering mode. Mipmap modes use UV derivatives "
            "so each screen pixel propagates gradients to a wider UV footprint."
        ),
    )
    # LoRA 秩
    parser.add_argument("--rank", type=int, default=64)
    # parser.add_argument("--rank_alpha", type=int, default=64)
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
    parser.add_argument(
        "--lora_last_steps",
        "--lora_last_t",
        dest="lora_last_steps",
        type=int,
        default=None,
        help=(
            "Only enable LoRA during the final N denoising steps. "
            "Default keeps LoRA active for all denoising steps. "
            "Example: --lora_last_steps 4 with --num_inference_steps 20."
        ),
    )
    parser.add_argument("--clip_weight", type=float, default=0.2)
    # masked-CLIP：CLIP 只对 UV mask 内计分（mask 外填中性灰）。
    # 动机：reward 梯度支撑只在 mask 内（只有这些 texel 会被渲染上车），
    # 全图 CLIP 会被优化器"外包"到 mask 外 71% 的免费面积上，
    # 导致上车区域 prompt 忠实度为零（实测 mask 外 +0.018 / mask 内 -0.013）。
    parser.add_argument(
        "--clip_uv_masked",
        action="store_true",
        help="Compute the CLIP objective on the UV-masked texture (outside region "
        "filled with `clip_mask_fill`) so prompt-consistency is scored only on "
        "texels that actually get rendered onto the car.",
    )
    parser.add_argument(
        "--clip_mask_fill",
        type=float,
        default=0.5,
        help="Fill value outside the UV mask when --clip_uv_masked is on. "
        "Neutral gray 0.5 preserves CLIP's pattern discrimination best.",
    )
    # 学习率
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    # 梯度裁剪上限
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    # 每隔多少个优化步记录一次各 loss 对 LoRA 参数的梯度范数；<=0 表示关闭
    parser.add_argument("--grad_norm_logging_steps", type=int, default=0)
    # 颜色分布 anchor（软直方图）：只锁颜色分布、放开结构。per-prompt 锚 base 纹理。
    parser.add_argument(
        "--color_weight",
        type=float,
        default=0.0,
        help="颜色分布 anchor 权重（0=关）。约束 LoRA 纹理颜色分布贴近 base。",
    )
    parser.add_argument(
        "--color_budget",
        type=float,
        default=0.0,
        help="δ_color（W1 尺度≈归一化色值平移量）。dev≤δ 免罚（允许小改颜色），"
        "超出才罚。0=硬匹配。建议 0.05~0.15。",
    )
    parser.add_argument("--color_hist_bins", type=int, default=32)
    parser.add_argument(
        "--color_hist_sigma",
        type=float,
        default=0.05,
        help="软直方图高斯分配宽度（[0,1] 色值尺度）。",
    )
    # 跨 prompt 多样性排斥(memory-bank)：把当前纹理推离历史【不同 prompt】的
    # CLIP-image 表征。默认关(weight=0)，关时完全不算 CLIP-image forward、零开销。
    # ⚠️ 与攻击直接对抗(崩塌结构常是攻击最优)，margin/weight 要保守。
    parser.add_argument(
        "--diversity_weight",
        type=float,
        default=0.0,
        help="多样性排斥权重(0=关，关时零开销)。",
    )
    parser.add_argument(
        "--diversity_margin",
        type=float,
        default=0.6,
        help="cosine 相似度阈值，sim>margin(太像)才推。崩塌时 sim≈0.9+。",
    )
    parser.add_argument("--diversity_bank_size", type=int, default=64)
    parser.add_argument(
        "--init_lora_path",
        type=str,
        default=None,
        help=(
            "可选：从已保存的 LoRA checkpoint 目录热启动（含 "
            "pytorch_lora_weights.safetensors）。诊断用途：在已崩塌的 "
            "checkpoint 上继续训练/测梯度冲突。"
        ),
    )
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
    parser.add_argument("--checkpointing_steps", type=int, default=100)
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
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    # 扩散生成分辨率
    parser.add_argument("--resolution", type=int, default=640)
    # 渲染图像分辨率
    parser.add_argument("--render_image_size", type=int, default=640)
    # UV 纹理图分辨率
    parser.add_argument("--texture_image_size", type=int, default=640)
    # tokenizer 最大长度
    parser.add_argument("--tokenizer_max_length", type=int, default=None)
    # 文本编码时是否传 attention mask
    parser.add_argument("--text_encoder_use_attention_mask", action="store_true")
    # 目标类别 ID
    parser.add_argument("--target_class_id", type=int, default=None)
    # YOLO 奖励模型路径
    parser.add_argument("--reward_model_path", type=str, default="yolomodel/yolov3u.pt")
    parser.add_argument("--reward_mode", type=str, default="non_targeted")
    # reward 项的损失权重
    parser.add_argument("--reward_weight", type=float, default=1.0)
    # LoRA 正则项权重
    parser.add_argument("--lora_reg_weight", type=float, default=1e-4)
    # 负面提示词
    parser.add_argument("--negative_prompt", type=str, default="")
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
    # ----- 多轴 prompt pool 配置（对应论文 §3.2） -----
    # 是否启用多轴组合 prompt 池；False 时退回旧的 ALL_PROMPTS 路径
    parser.add_argument("--use_prompt_pool", action="store_true")
    parser.add_argument("--use_prompt_pool_codex", action="store_true")
    # 中性锚点 prompt 出现概率（prior preservation 替代项）
    parser.add_argument("--prompt_p_neutral", type=float, default=0.10)
    # prompt pool 划分种子（控制 train/comp/ood 划分，复现关键）
    parser.add_argument("--prompt_pool_seed", type=int, default=42)
    # 每个轴预留多少值作 OOD 测试集
    parser.add_argument("--prompt_holdout_per_axis", type=int, default=2)
    # 预留多少个 (material, context) 对作组合泛化测试集
    parser.add_argument("--prompt_holdout_combos", type=int, default=8)
    # 渲染所用 OBJ 模型路径
    parser.add_argument(
        "--mesh_obj_path", type=str, default="carmodel/audi/pytorch3d_Etron.obj"
    )
    parser.add_argument(
        "--use_carla_mesh_transform",
        action="store_true",
        help="Use CARLA-aligned mesh centering, scaling, and yaw rotation.",
    )
    # UV 掩码图片路径
    parser.add_argument(
        "--uv_mask_path", type=str, default="carmodel/audi/modified_mask.png"
    )
    # UV 掩码外区域的填充值
    parser.add_argument("--background_fill_value", type=float, default=0.0)
    parser.add_argument(
        "--bptt",
        type=int,
        nargs="+",
        default=[18],
        help=(
            "Candidate truncation step indices for BPTT. One value is sampled "
            "each denoising run; e.g. '--bptt 15 16' with 20 inference steps "
            "updates the last 5 or 4 denoising steps."
        ),
    )

    parser.add_argument("--use_diffusion_image_aug", action="store_true")
    parser.add_argument("--use_roa", action="store_true")
    parser.add_argument("--ddim_eta", default=0.0, type=float)
    parser.add_argument("--renderer_backend", default="pytorch3d", type=str)

    parser.add_argument(
        "--traj_anchor_weight",
        type=float,
        default=0.0,
        help="Weight for trajectory anchor loss (epsilon consistency). 0 disables.",
    )
    parser.add_argument(
        "--traj_anchor_t_min",
        type=int,
        default=400,
        help="Lower bound (inclusive) of timestep range for trajectory anchor. "
        "Should sit comfortably above the BPTT window so structure-step anchoring "
        "does not collide with texture-step adversarial backprop.",
    )
    parser.add_argument(
        "--traj_anchor_t_max",
        type=int,
        default=999,
        help="Upper bound (inclusive) of timestep range for trajectory anchor.",
    )
    parser.add_argument(
        "--traj_anchor_prob",
        type=float,
        default=1.0,
        help="Probability of applying the trajectory anchor at each adversarial step. "
        "Lower this (e.g. 0.5) to amortize the extra UNet forward+backward across steps.",
    )
    parser.add_argument(
        "--traj_anchor_budget",
        type=float,
        default=0.0,
        help="eps-deviation budget δ (RMS units) for the trajectory anchor. "
        "0 → hard anchor (drive eps_on→eps_off, 0 扰动=不许攻击). "
        ">0 → soft budget: 只惩罚 RMS 偏离超过 δ 的部分 (relu(dev-δ)²)，"
        "允许在 base 轨迹周围换姿势/花样、只禁止改出类。δ 是攻击强度↔类内保真的旋钮。"
        "配合门控时锚点自动落在 LoRA 激活的门控窗口 (t 见 --lora_last_steps)。",
    )

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
    if args.lora_last_steps is not None and (
        args.lora_last_steps < 1 or args.lora_last_steps > args.num_inference_steps
    ):
        raise ValueError(
            "`lora_last_steps` must be in [1, num_inference_steps], "
            f"got {args.lora_last_steps} for num_inference_steps={args.num_inference_steps}."
        )
    if not args.bptt:
        raise ValueError("`bptt` must contain at least one truncation step.")
    for bptt_step in args.bptt:
        if bptt_step < 0 or bptt_step >= args.num_inference_steps:
            raise ValueError(
                "`bptt` values must be in [0, num_inference_steps - 1], "
                f"got {bptt_step} for num_inference_steps={args.num_inference_steps}."
            )
    if args.resolution % 8 != 0:
        raise ValueError("`resolution` must be divisible by 8 for SDXL latent scaling.")
    if not 0.0 <= args.background_fill_value <= 1.0:
        raise ValueError("`background_fill_value` must be in [0, 1].")

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



def cycle_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def render_reward_multi_view_carla(
    texture_render_model,
    yolo_reward_model,
    diffusion_images: torch.Tensor,
    carla_batch,
    carla_dataset,
    num_views: int,
    reward_mode: str = "non_targeted",
    target_class_id: int = 2,
    roa=None,
):
    rendered_views = []
    reward_values = []
    reward_statuses = []
    view_cameras = []

    device = diffusion_images.device
    carla_images = carla_batch["image"].to(device)
    batch_size = carla_images.shape[0]
    views_to_render = min(num_views, batch_size)
    if views_to_render < 1:
        raise ValueError("CARLA batch is empty; cannot render reward views.")

    for view_index in range(views_to_render):
        cam = build_cameras(
            carla_batch["R"][view_index : view_index + 1],
            carla_batch["T"][view_index : view_index + 1],
            carla_dataset,
            device=device,
        )
        view_background = carla_images[view_index : view_index + 1]
        rendered = texture_render_model(cam, diffusion_images, view_background)
        rendered = rendered.permute(0, 3, 1, 2).clamp(0, 1)
        if roa is not None:
            rendered = roa(rendered)
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
    )


def loss_lora_grad_norm(
    loss: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
    accelerator: Accelerator,
) -> torch.Tensor:
    device = loss.device if loss.is_cuda else accelerator.device
    total_sq = torch.zeros((), device=device, dtype=torch.float32)
    if not loss.requires_grad:
        return total_sq

    grads = torch.autograd.grad(
        loss,
        [param for param in params if param.requires_grad],
        retain_graph=True,
        allow_unused=True,
    )
    for grad in grads:
        if grad is None:
            continue
        total_sq = total_sq + grad.detach().float().pow(2).sum()

    if accelerator.num_processes > 1:
        total_sq = accelerator.reduce(total_sq, reduction="sum")
    return total_sq.sqrt()


def loss_lora_grad_flat(
    loss: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
) -> Optional[torch.Tensor]:
    """
    单个损失对 LoRA 参数的完整梯度，展平成一个 fp32 向量（detached）。

    用于两两余弦相似度诊断：cos<0 = 方向冲突（PCGrad 的适用域），
    cos≈0 但范数差几个数量级 = 量级失衡（该归一化/GradNorm，PCGrad 无效）。
    注意：单进程语义（多卡下各 rank 只看到本地梯度）。
    """
    if not loss.requires_grad:
        return None
    req = [p for p in params if p.requires_grad]
    grads = torch.autograd.grad(loss, req, retain_graph=True, allow_unused=True)
    return torch.cat(
        [
            (g if g is not None else torch.zeros_like(p)).detach().float().reshape(-1)
            for g, p in zip(grads, req)
        ]
    )


def unwrap_model(accelerator: Accelerator, model):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def get_adapter_state_model(model):
    model = getattr(model, "module", model)
    return model._orig_mod if is_compiled_module(model) else model


def prepare_scheduler_ddim(base_scheduler, num_inference_steps: int):
    scheduler = DDIMScheduler.from_config(base_scheduler.config)
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
    args=None,
    disable_lora_all: bool = False,
):
    """
    执行 DDIM 采样，并在截断点 K 之后保留梯度计算图。

    - step_idx < K（早期去噪步）：stop_gradient，不参与反向传播
    - step_idx >= K（后期去噪步）：保留计算图，梯度可以流过

    K 是 denoising step index。20 步推理时，K=15 表示只更新最后 5 步，
    K=19 表示只更新最后 1 步。若 args.bptt 传入多个候选值，每次采样随机选一个。

    返回 (images, anchor_states)：anchor_states 是本次去噪真实经过的
    (x_t.detach(), t) 列表（凡 LoRA 激活的步都收，不在此按 t 过滤）。
    t 窗口 [traj_anchor_t_min, traj_anchor_t_max] 的过滤在消费端做（选锚点时），
    供 on-trajectory anchor 使用——锚在生成真正访问过的状态上，而非随机高斯。
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

    # 采样截断点 K：从第 K 步开始才保留梯度
    total_steps = len(scheduler.timesteps)
    if args is not None and hasattr(args, "bptt"):
        bptt_candidates = args.bptt
        if isinstance(bptt_candidates, int):
            bptt_candidates = [bptt_candidates]
        K = random.choice(list(bptt_candidates))
    elif truncated_backprop_rand:
        K = random.randint(
            truncated_backprop_minmax[0],
            min(truncated_backprop_minmax[1], total_steps - 1),
        )
    else:
        K = min(truncated_backprop_minmax[1], total_steps - 1)
    # print(f"k={K}")
    eta = args.ddim_eta
    lora_last_steps = (
        getattr(args, "lora_last_steps", None) if args is not None else None
    )
    lora_start_step = 0 if lora_last_steps is None else total_steps - lora_last_steps
    adapter_state_model = get_adapter_state_model(unet)

    # on-trajectory anchor：缓存 LoRA 真正激活的步真实经过的 (x_t, t)。
    # x_t 是 UNet 在该步收到的输入（detach 后仅 ~0.05MB/个）。
    # 关键：锚点必须落在 LoRA 激活处——门控 (--lora_last_steps) 下 LoRA 只在最后
    # 几步 (低 t) 激活，此时 [traj_anchor_t_min,max]=[400,999] 完全避开门控窗口，
    # 会导致锚点全为空/恒 0。故这里按"LoRA 是否激活"收集，而非按固定 t 区间。
    # eps-budget 的 hinge 会自动只在偏离超预算的步上产生梯度，无需再按 t 精选。
    anchor_states = []

    for step_idx, timestep in enumerate(scheduler.timesteps):
        t_int = int(timestep)
        # disable_lora_all: 全程关 LoRA，用于生成 base 纹理（颜色 anchor 的锚点）。
        lora_active = (not disable_lora_all) and (step_idx >= lora_start_step)
        lora_context = (
            nullcontext() if lora_active else disable_lora_context(adapter_state_model)
        )
        if lora_active:
            anchor_states.append((latents.detach(), t_int))
        # step_idx < K：截断，不保留梯度（减少显存，防止 mode collapse）
        # step_idx >= K：保留梯度，参与 reward backprop
        if step_idx < K:
            with torch.no_grad():
                with lora_context:
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
                        eta,
                    )
                latents = latents.detach()  # 显式断开计算图
        else:
            # gradient checkpointing 在外部通过 unet.enable_gradient_checkpointing() 控制
            with lora_context:
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
                    eta,
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

    return images, anchor_states


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
    eta=0.0,
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
    # 关键修改：unconditional CFG 分支不接收检测器 reward 梯度

    noise_pred_uncond = noise_pred_uncond.detach()

    guided_noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )
    guided_noise_pred = guided_noise_pred.to(dtype=dtype)

    latents = scheduler.step(
        guided_noise_pred, timestep, latents, eta=eta, return_dict=False
    )[0]
    return latents.to(dtype=dtype)


def get_base_color_hist(
    cache: dict,
    prompt_key: str,
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
    resolution: int,
    guidance_scale: float,
    args,
    uv_mask,
    bins: int,
    sigma: float,
):
    """
    取（或惰性生成并缓存）某 prompt 的 base 纹理颜色直方图。

    base = 全程关 LoRA 采样出的纹理（disable_lora_all=True）。颜色分布对
    初始噪声不敏感（同 prompt 同色调），故每 prompt 只生成一次、缓存直方图
    （几十个数），后续复用。key 用 prompt 字符串。
    """
    if prompt_key in cache:
        return cache[prompt_key]
    with torch.no_grad():
        base_img, _ = sample_latents_with_grad(
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
            1,
            resolution,
            guidance_scale,
            args=args,
            disable_lora_all=True,
        )
        hist = soft_color_histogram(base_img, uv_mask, bins, sigma).detach()
    cache[prompt_key] = hist
    return hist


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


def main(args):
    ensure_required_paths(args)

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

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
    temp_args = {
        key: str(value)
        if value is None or isinstance(value, (list, tuple, dict))
        else value
        for key, value in temp_args.items()
    }

    accelerator.init_trackers(tracker_name, config=temp_args)

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
    if args.reward_model_path == "yolomodel/yolov3lz.pt":
        from thirdpart.yolov3.yolov3_detector import YoloV3Detector

        yolo_reward_model = YoloV3Detector(
            args.reward_model_path, device=accelerator.device
        )
    else:
        from utils.yolo import YoloRewardModel

        yolo_reward_model = YoloRewardModel(
            args.reward_model_path,
        )
    # 加载clip模型
    clip_model = CLIPModel.from_pretrained(args.clip_path)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_path)
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
    if args.use_roa:
        roa = ROA()  # 实例化 ROA
        roa.to(accelerator.device)
    else:
        roa = None

    roa_diffusion_image = ROADiffusionImage()
    roa_diffusion_image.to(accelerator.device)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=vae_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    yolo_reward_model.to(accelerator.device)
    clip_model.to(accelerator.device, dtype=weight_dtype)

    uv_mask = load_uv_mask(
        args.uv_mask_path, args.texture_image_size, accelerator.device
    )
    if args.renderer_backend == "nvdiffrast":
        from meshrender.nvdiffrast_uv import NvdiffrastTextureRenderModel

        texture_render_model = NvdiffrastTextureRenderModel(
            device=accelerator.device,
            mesh_obj_path=args.mesh_obj_path,
            uv_mask=uv_mask,
            render_image_size=args.render_image_size,
            texture_image_size=args.texture_image_size,
            background_fill_value=args.background_fill_value,
            use_carla_mesh_transform=args.use_carla_mesh_transform,
            # use_antialias=True,
            texture_filter_mode=args.nvdiffrast_texture_filter,
        )
    else:
        texture_render_model = TextureRenderModel(
            device=accelerator.device,
            mesh_obj_path=args.mesh_obj_path,
            uv_mask=uv_mask,
            render_image_size=args.render_image_size,
            texture_image_size=args.texture_image_size,
            background_fill_value=args.background_fill_value,
            use_carla_mesh_transform=args.use_carla_mesh_transform,
            raster_blur_radius=0,
            raster_faces_per_pixel=1,
        )
    texture_render_model.to(accelerator.device)

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
        # use_rslora=args.use_rslora,
        # target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
        target_modules=lora_target_modules,
    )
    unet.add_adapter(unet_lora_config)
    cast_training_params([unet], dtype=torch.float32)

    if args.init_lora_path:
        # 从 save_lora_weights 保存的 checkpoint 热启动（诊断/续训用）。
        from diffusers.utils import convert_unet_state_dict_to_peft
        from peft.utils import set_peft_model_state_dict

        lora_state, _ = StableDiffusionXLPipeline.lora_state_dict(args.init_lora_path)
        unet_state = {
            k.replace("unet.", "", 1): v
            for k, v in lora_state.items()
            if k.startswith("unet.")
        }
        unet_state = convert_unet_state_dict_to_peft(unet_state)
        incompatible = set_peft_model_state_dict(
            unet, unet_state, adapter_name="default"
        )
        unexpected = getattr(incompatible, "unexpected_keys", [])
        if unexpected:
            raise ValueError(
                f"init_lora_path 加载有 unexpected keys（scope/rank 不匹配？）: "
                f"{unexpected[:5]}..."
            )
        cast_training_params([unet], dtype=torch.float32)
        logger.info("Loaded init LoRA weights from %s", args.init_lora_path)

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
    if args.lora_last_steps is None:
        logger.info("LoRA is enabled for all denoising steps.")
    else:
        logger.info(
            "LoRA is enabled only for the final %d/%d denoising steps.",
            args.lora_last_steps,
            args.num_inference_steps,
        )

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
    from data.dataset import make_loader
    if args.use_old_data:
        root = [
            "data/datas/origin/si54/sunny/6m",
            "data/datas/origin/si54/sunny/8m",
            "data/datas/origin/si54/sunny/10m",
            "data/datas/origin/si54/sunny/12m",
            "data/datas/origin/si139/sunny/6m",
            "data/datas/origin/si139/sunny/8m",
            "data/datas/origin/si139/sunny/10m",
            "data/datas/origin/si139/sunny/12m",
        ]
    else:
        root = [
            "data/datas/origin/si103/day_clear/8m",
            "data/datas/origin/si103/day_clear/10m",
            "data/datas/origin/si103/day_clear/12m",
            "data/datas/origin/si103/day_cloudy/8m",
            "data/datas/origin/si103/day_cloudy/10m",
            "data/datas/origin/si103/day_cloudy/12m",
            "data/datas/origin/si103/night_clear/8m",
            "data/datas/origin/si103/night_clear/10m",
            "data/datas/origin/si103/night_clear/12m",
            "data/datas/origin/si103/night_cloudy/8m",
            "data/datas/origin/si103/night_cloudy/10m",
            "data/datas/origin/si103/night_cloudy/12m",
            "data/datas/origin/si139/sunny/6m",
            "data/datas/origin/si139/sunny/8m",
            "data/datas/origin/si139/sunny/10m",
            "data/datas/origin/si139/sunny/12m",
            "data/datas/origin/si47/day_clear/8m",
            "data/datas/origin/si47/day_clear/10m",
            "data/datas/origin/si47/day_clear/12m",
            "data/datas/origin/si47/day_cloudy/8m",
            "data/datas/origin/si47/day_cloudy/10m",
            "data/datas/origin/si47/day_cloudy/12m",
            "data/datas/origin/si47/night_clear/8m",
            "data/datas/origin/si47/night_clear/10m",
            "data/datas/origin/si47/night_clear/12m",
            "data/datas/origin/si47/night_cloudy/8m",
            "data/datas/origin/si47/night_cloudy/10m",
            "data/datas/origin/si47/night_cloudy/12m",
            "data/datas/origin/si54/sunny/6m",
            "data/datas/origin/si54/sunny/8m",
            "data/datas/origin/si54/sunny/10m",
            "data/datas/origin/si54/sunny/12m",
            "data/datas/origin/si70/day_clear/8m",
            "data/datas/origin/si70/day_clear/10m",
            "data/datas/origin/si70/day_clear/12m",
            "data/datas/origin/si70/day_cloudy/8m",
            "data/datas/origin/si70/day_cloudy/10m",
            "data/datas/origin/si70/day_cloudy/12m",
            "data/datas/origin/si70/night_clear/8m",
            "data/datas/origin/si70/night_clear/10m",
            "data/datas/origin/si70/night_clear/12m",
            "data/datas/origin/si70/night_cloudy/8m",
            "data/datas/origin/si70/night_cloudy/10m",
            "data/datas/origin/si70/night_cloudy/12m",
            # "data/datas/origin/si75/sunny/8m",
            # "data/datas/origin/si75/sunny/10m",
            # "data/datas/origin/si75/sunny/12m",
        ]
    carla_ds, carla_loader = make_loader(
        root=root,
        batch_size=args.num_reward_views,
        shuffle=True,
        num_workers=4,
    )
    unet, optimizer, lr_scheduler, carla_loader = accelerator.prepare(
        unet, optimizer, lr_scheduler, carla_loader
    )

    global_step = 0
    # per-prompt base 纹理颜色直方图缓存（颜色分布 anchor 用，惰性填充）
    base_color_hist_cache = {}
    # 多样性排斥 memory bank：存 (prompt, clip_img_feat.detach())。默认关时不使用。
    diversity_bank = deque(maxlen=args.diversity_bank_size)
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
    logger.info("Total selected prompts (legacy pool): %d", len(total_prompts))

    prompt_pool: Optional[PromptPool, PromptPoolCodex] = None

    if args.use_prompt_pool:
        prompt_pool = PromptPool(
            seed=args.prompt_pool_seed,
            p_neutral=args.prompt_p_neutral,
            holdout_per_axis=args.prompt_holdout_per_axis,
            holdout_combos=args.prompt_holdout_combos,
        )
        if accelerator.is_main_process:
            split_path = Path(args.output_dir) / "prompt_split.json"
            prompt_pool.save_split(split_path)
            logger.info(
                "Prompt pool initialized (seed=%d, p_neutral=%.2f). Split saved to %s",
                args.prompt_pool_seed,
                args.prompt_p_neutral,
                split_path,
            )
            desc = prompt_pool.describe()
            logger.info(
                "Train values per axis: %s",
                {k: len(v) for k, v in desc["train_values"].items()},
            )
            logger.info("OOD values per axis : %s", desc["ood_values"])
            logger.info("Held-out comp pairs : %d", len(desc["comp_pairs"]))

    if args.use_prompt_pool_codex:
        prompt_pool = PromptPoolCodex(
            seed=args.prompt_pool_seed,
            p_neutral=args.prompt_p_neutral,
            holdout_per_axis=args.prompt_holdout_per_axis,
            holdout_combos=args.prompt_holdout_combos,
        )
    carla_iter = cycle_dataloader(carla_loader)
    idx = 0
    while global_step < args.max_train_steps:
        batch = next(carla_iter)
        scheduler = prepare_scheduler_ddim(base_scheduler, args.num_inference_steps)

        if prompt_pool is not None:
            pool_sample = prompt_pool.sample_train()
            prompts = [pool_sample.text]
        else:
            # 兼容旧路径：单 prompt + 颜色拼接
            prompts = [random.choice(total_prompts)]
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
            diffusion_images, traj_anchor_states = sample_latents_with_grad(
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
                args=args,
            )
            clip_need_grad = True
            if args.clip_weight == 0:
                clip_need_grad = False
            # masked-CLIP：只在 UV mask 内计分。mask 乘法把 mask 外像素的 CLIP 梯度
            # 归零，使 CLIP 梯度支撑与 reward 梯度支撑重合，逼模型在"上车纹理"上兼顾
            # prompt 与生成的纹理，而不是把 prompt 一致性外包给 mask 外的免费面积。
            if args.clip_uv_masked:
                clip_mask = uv_mask.unsqueeze(0).to(dtype=diffusion_images.dtype)
                if clip_mask.shape[-2:] != diffusion_images.shape[-2:]:
                    clip_mask = torch.nn.functional.interpolate(
                        clip_mask, size=diffusion_images.shape[-2:], mode="nearest"
                    )
                clip_input = diffusion_images * clip_mask + args.clip_mask_fill * (
                    1 - clip_mask
                )
            else:
                clip_input = diffusion_images
            clip_score = clip_text_image_similarity(
                clip_processor,
                clip_model,
                clip_input,
                prompts,
                accelerator.device,
                needgrad=clip_need_grad,
            )
            # 全图 CLIP：仅用于日志，便于与历史(未 mask) run 对照；不进 loss、不回传。
            if args.clip_uv_masked:
                with torch.no_grad():
                    clip_score_full = clip_text_image_similarity(
                        clip_processor,
                        clip_model,
                        diffusion_images,
                        prompts,
                        accelerator.device,
                        needgrad=False,
                    )
            else:
                clip_score_full = clip_score.detach()
            # 颜色分布 anchor：只锁颜色、放开结构。用未增强的 LoRA-on 纹理图，
            # 逐 prompt 锚各自 base 纹理的颜色直方图（惰性生成缓存）。
            if args.color_weight > 0:
                base_hists = []
                for i, prompt_key in enumerate(prompts):
                    base_hists.append(
                        get_base_color_hist(
                            base_color_hist_cache,
                            prompt_key,
                            unet,
                            vae,
                            scheduler,
                            prompt_embeds[i : i + 1],
                            pooled_prompt_embeds[i : i + 1],
                            negative_prompt_embeds[i : i + 1],
                            negative_pooled_prompt_embeds[i : i + 1],
                            add_time_ids[i : i + 1],
                            accelerator.device,
                            weight_dtype,
                            args.resolution,
                            args.guidance_scale,
                            args,
                            uv_mask,
                            args.color_hist_bins,
                            args.color_hist_sigma,
                        )
                    )
                base_hist = torch.cat(base_hists, dim=0)
                loss_color_raw, color_dev = color_distribution_loss(
                    diffusion_images,
                    base_hist,
                    uv_mask,
                    args.color_hist_bins,
                    args.color_hist_sigma,
                    budget=args.color_budget,
                )
                loss_color = args.color_weight * loss_color_raw
            else:
                loss_color = torch.zeros(
                    (), device=accelerator.device, dtype=weight_dtype
                )
                color_dev = torch.zeros((), device=accelerator.device)
            # 跨 prompt 多样性排斥(memory-bank)：把当前纹理推离历史【不同 prompt】的
            # CLIP-image 表征。默认关(weight=0)——关时完全不算 CLIP-image forward。
            if args.diversity_weight > 0:
                cur_div_feat = clip_image_feature(
                    clip_processor,
                    clip_model,
                    diffusion_images,
                    accelerator.device,
                    needgrad=True,
                )  # (B, D) 归一化，有梯度
                cur_prompt_key = prompts[0]
                other_feats = [f for pk, f in diversity_bank if pk != cur_prompt_key]
                bank_feats = torch.cat(other_feats, dim=0) if other_feats else None
                loss_div_raw, div_max_sim = diversity_repulsion_loss(
                    cur_div_feat, bank_feats, margin=args.diversity_margin
                )
                loss_div = args.diversity_weight * loss_div_raw
                diversity_bank.append((cur_prompt_key, cur_div_feat.detach()))
            else:
                loss_div = torch.zeros(
                    (), device=accelerator.device, dtype=weight_dtype
                )
                div_max_sim = torch.zeros((), device=accelerator.device)
            if args.use_diffusion_image_aug:
                diffusion_images = roa_diffusion_image(diffusion_images)

            reward_outputs = render_reward_multi_view_carla(
                texture_render_model=texture_render_model,
                yolo_reward_model=yolo_reward_model,
                diffusion_images=diffusion_images,
                carla_batch=batch,
                carla_dataset=carla_ds,
                num_views=args.num_reward_views,
                reward_mode=args.reward_mode,
                target_class_id=2,
                roa=roa,
            )
            rendered = reward_outputs["rendered_views"]
            rewardStatus = reward_outputs["reward_status"]
            reward_value = reward_outputs["reward_value"]
            reward_values = reward_outputs["reward_values"]

            # 每20步保存一次渲染结果
            if accelerator.is_main_process and global_step % 100 == 0:
                # if accelerator.is_main_process and reward_value< 0.05:
                save_path = Path(args.output_dir) / "save" / str(global_step)
                save_path.mkdir(parents=True, exist_ok=True)
                for view_idx in range(len(rendered)):
                    rendered_image = tensor_rgb_to_pil(rendered[view_idx])
                    rendered_image.save(f"{save_path}/rendered_view{view_idx}.png")
            if accelerator.is_main_process and global_step % 20 == 0:
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
                rendered_image = tensor_rgb_to_pil(rendered[0])
                texture_image = tensor_rgb_to_pil(diffusion_images)

                # rendered_image.save(
                #     f"{args.output_dir}/{global_step}_rendered_images_{idx}.png"
                # )
                # texture_image.save(
                #     f"{args.output_dir}/{global_step}_texture_images_{idx}.png"
                # )

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
                    f"{args.output_dir}/{global_step}_combined_images_{idx}.png"
                )
                log_prompt_file = f"{args.output_dir}/log.txt"
                with open(log_prompt_file, "a+") as f:
                    f.write(f"{global_step}_{prompts}\n")
                    f.flush()

            # clip正则化项
            loss_clip_score = args.clip_weight * (1 - clip_score)  # -1,1 -> 0,2,
            loss_reg = args.lora_reg_weight * lora_regularization(unet)
            loss_reward = args.reward_weight * reward_value
            loss_reward_active = (
                loss_reward if rewardStatus else torch.zeros_like(loss_reward)
            )
            loss_reward_values_active = (
                reward_values if rewardStatus else torch.zeros_like(loss_reward)
            )
            if not rewardStatus:
                logger.info(
                    f"Step {global_step}: 攻击成功, CLIP Score:   {clip_score.item():.4f}"
                )

            if args.use_margin_loss:
                loss_reward_active2, frac_below_margin = margin_detection_loss(
                    # loss_reward_active,  # (V,) per-view tensor，现在派上用场
                    loss_reward_values_active,  # (V,) per-view tensor，现在派上用场
                    margin=args.margin,
                    loss_type="squared_hinge",
                    # loss_type="softplus",
                )
            else:
                loss_reward_active2 = loss_reward_values_active.mean()
                frac_below_margin = torch.Tensor([-1.0])

            reward_loss = loss_reward_active2 * args.reward_weight

            # ------------------------------------------------------
            should_apply_traj_anchor = (
                args.traj_anchor_weight > 0 and random.random() < args.traj_anchor_prob
            )
            # on-trajectory anchor：从本次真实去噪轨迹里随机挑一个 (x_t, t) 作锚点，
            # 约束"生成真正经过的状态"上 LoRA ≈ base。
            # 关键：只在结构区 [t_min, t_max] 内选锚点，避开低 t 的 BPTT 攻击窗口。
            # 否则锚会落在攻击真正干活的低 t 纹理步上，直接罚掉攻击本身（eps 偏离
            # 在低 t = 攻击信号，同一 δ 在这里几乎恒触发 hinge，把正则变成反攻击力）。
            # 窗口内无状态时回退 None → 函数内随机高斯 + randint(t_min,t_max) 兜底。
            anchor_pool = [
                (z, t)
                for (z, t) in traj_anchor_states
                if args.traj_anchor_t_min <= t <= args.traj_anchor_t_max
            ]
            if anchor_pool:
                anchor_z_t, anchor_timestep = random.choice(anchor_pool)
            else:
                anchor_z_t, anchor_timestep = None, None

            if args.use_old_anchor:
                anchor_z_t, anchor_timestep = None, None
                
            if should_apply_traj_anchor:
                loss_traj_hinge, loss_traj_dev = trajectory_anchor_loss(
                    unwrap_model(accelerator, unet),
                    prompt_embeds,
                    pooled_prompt_embeds,
                    add_time_ids,
                    args.resolution,
                    accelerator.device,
                    weight_dtype,
                    t_min=args.traj_anchor_t_min,
                    t_max=args.traj_anchor_t_max,
                    z_t=anchor_z_t,
                    timesteps=anchor_timestep,
                    budget=args.traj_anchor_budget,
                )
                loss_traj_raw = loss_traj_hinge
                loss_traj = args.traj_anchor_weight * loss_traj_hinge
            else:
                with torch.no_grad():
                    loss_traj_raw, loss_traj_dev = trajectory_anchor_loss(
                        unwrap_model(accelerator, unet),
                        prompt_embeds,
                        pooled_prompt_embeds,
                        add_time_ids,
                        args.resolution,
                        accelerator.device,
                        weight_dtype,
                        t_min=args.traj_anchor_t_min,
                        t_max=args.traj_anchor_t_max,
                        z_t=anchor_z_t,
                        timesteps=anchor_timestep,
                        budget=args.traj_anchor_budget,
                    )
                loss_traj = torch.zeros(
                    (), device=accelerator.device, dtype=weight_dtype
                )

            loss = (
                reward_loss
                + loss_clip_score
                + loss_reg
                + loss_traj
                + loss_color
                + loss_div
            )
            # loss = loss_traj
            # 清理无用的计算图部分，节省显存
            if (
                not rewardStatus
                and args.clip_weight == 0
                and args.traj_anchor_weight == 0
                and args.lora_reg_weight == 0
            ):
                global_step += 1
                idx += 1
                logs = {
                    "loss": loss.detach().item(),
                    "loss_reward": reward_loss.detach().item(),
                    "loss_reward_raw": loss_reward_active.detach().item(),
                    "loss_reg": loss_reg.detach().item(),
                    "loss_clip_score": loss_clip_score.detach().item(),
                    "loss_traj": loss_traj.detach().item(),
                    "loss_traj_raw": loss_traj_raw.detach().item(),
                    "loss_traj_dev": loss_traj_dev.detach().item(),
                    "loss_color": loss_color.detach().item(),
                    "color_dev": float(color_dev),
                    "loss_div": loss_div.detach().item(),
                    "div_max_sim": float(div_max_sim),
                    "traj_on": float(should_apply_traj_anchor),
                    "clip_score": clip_score.detach().item(),
                    "clip_score_full": clip_score_full.item(),
                    "reward": reward_value.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "FracBelowMargin": frac_below_margin.detach().item(),
                }
                accelerator.log(logs, step=global_step)
                optimizer.zero_grad(set_to_none=True)
                del (
                    loss_reward_active,
                    loss_reward,
                    reward_outputs,
                    rendered,
                    reward_value,
                    reward_values,
                    diffusion_images,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    add_time_ids,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue  # reward太小不更新了，节省计算
            grad_norm_logs = {}
            should_log_grad_norms = (
                args.grad_norm_logging_steps > 0
                and accelerator.sync_gradients
                and global_step % args.grad_norm_logging_steps == 0
            )
            if should_log_grad_norms:
                # 每个损失单独的 LoRA 梯度向量（fp32, detached），测范数 + 两两余弦。
                # 余弦判"方向冲突"(PCGrad 适用)，范数比判"量级失衡"(GradNorm 适用)。
                grad_vecs = {
                    "reward": loss_lora_grad_flat(reward_loss, params_to_optimize),
                    "clip": loss_lora_grad_flat(loss_clip_score, params_to_optimize),
                    "reg": loss_lora_grad_flat(loss_reg, params_to_optimize),
                    "traj": loss_lora_grad_flat(loss_traj, params_to_optimize)
                    if should_apply_traj_anchor
                    else None,
                }
                grad_norms = {
                    name: (vec.norm().item() if vec is not None else 0.0)
                    for name, vec in grad_vecs.items()
                }
                grad_norm_logs = {
                    f"grad_norm_{name}_lora": value
                    for name, value in grad_norms.items()
                }
                # 兼容旧日志键名
                reward_grad_norm_item = grad_norms["reward"]
                clip_grad_norm_item = grad_norms["clip"]
                grad_cos_pairs = [
                    ("reward", "clip"),
                    ("reward", "traj"),
                    ("reward", "reg"),
                    ("clip", "traj"),
                ]
                grad_cos_logs = {}
                for name_a, name_b in grad_cos_pairs:
                    vec_a, vec_b = grad_vecs[name_a], grad_vecs[name_b]
                    if (
                        vec_a is None
                        or vec_b is None
                        or grad_norms[name_a] == 0.0
                        or grad_norms[name_b] == 0.0
                    ):
                        continue
                    cos = (vec_a @ vec_b) / (vec_a.norm() * vec_b.norm() + 1e-12)
                    grad_cos_logs[f"grad_cos_{name_a}_{name_b}"] = cos.item()
                grad_norm_logs.update(grad_cos_logs)
                del grad_vecs
            # end clip gate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            # optimizer.zero_grad(set_to_none=False)
            optimizer.zero_grad()
            if accelerator.is_main_process and global_step % 1 == 0:
                reward_values_text = ", ".join(
                    f"{value:.4f}" for value in reward_values.detach().cpu().tolist()
                )
                # reward_values_text = f"{reward_value.item():.4f}"
                logger.info(
                    f"Step {global_step}: Loss={loss.detach().item():.4f}, "
                    f"LossReward={reward_loss.detach().item():.4f}, "
                    f"LossReward_raw={loss_reward_active.detach().item():.4f}, "
                    f"LossColor={loss_color.detach().item():.6f}, "
                    f"ColorDev={float(color_dev):.4f}, "
                    f"LossDiv={loss_div.detach().item():.6f}, "
                    f"DivMaxSim={float(div_max_sim):.4f}, "
                    # f"LossReg={loss_reg.detach().item():.8f}, "
                    # f"LossClip={loss_clip_score.detach().item():.4f}, "
                    # f"LossTraj={loss_traj.detach().item():.8f}, "
                    # f"LossTraj_raw={loss_traj_raw.detach().item():.8f}, "
                    # f"TrajDev={loss_traj_dev.detach().item():.6f}, "
                    # f"TrajOn={int(should_apply_traj_anchor)}, "
                    # f"ClipScore={clip_score.detach().item():.4f}, "
                    # f"ClipScoreFull={clip_score_full.item():.4f}, "
                    # f"RewardMean={reward_value.detach().item():.4f}, "
                    # f"FracBelowMargin={frac_below_margin.detach().item():.3f}, "
                    # f"reward_values_text={reward_values_text}, "
                    # f"LR={lr_scheduler.get_last_lr()[0]:.6f}"
                )
                logger.info(f"Step {global_step}: Prompts: {prompts}")
                if grad_norm_logs:
                    cos_text = ", ".join(
                        f"cos({name_a},{name_b})={grad_cos_logs[f'grad_cos_{name_a}_{name_b}']:.4f}"
                        for name_a, name_b in grad_cos_pairs
                        if f"grad_cos_{name_a}_{name_b}" in grad_cos_logs
                    )
                    logger.info(
                        f"Step {global_step}: GradNormRewardLoRA={reward_grad_norm_item:.6e}, "
                        f"GradNormClipLoRA={clip_grad_norm_item:.6e}, "
                        f"GradNormTrajLoRA={grad_norms['traj']:.6e}, "
                        f"GradNormRegLoRA={grad_norms['reg']:.6e}, "
                        f"GradNormRewardClipRatio={reward_grad_norm_item / (clip_grad_norm_item + 1e-12):.6e}, "
                        f"cos:{cos_text}"
                    )

        if accelerator.sync_gradients:
            global_step += 1
            idx += 1
            logs = {
                "loss": loss.detach().item(),
                "loss_reward": reward_loss.detach().item(),
                "loss_reward_raw": loss_reward_active.detach().item(),
                "loss_reg": loss_reg.detach().item(),
                "loss_clip_score": loss_clip_score.detach().item(),
                "loss_traj": loss_traj.detach().item(),
                "loss_traj_raw": loss_traj_raw.detach().item(),
                "loss_color": loss_color.detach().item(),
                "color_dev": float(color_dev),
                "loss_div": loss_div.detach().item(),
                "div_max_sim": float(div_max_sim),
                "traj_on": float(should_apply_traj_anchor),
                "clip_score": clip_score.detach().item(),
                "ClipScoreFull": clip_score_full.item(),
                "reward": reward_value.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "FracBelowMargin": frac_below_margin.detach().item(),
            }
            logs.update(grad_norm_logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
            # 每 N次 梯度步保存一次 LoRA 权重检查点，并进行评估
            if (
                accelerator.is_main_process
                and global_step % args.checkpointing_steps == 0
                and global_step > 500
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
