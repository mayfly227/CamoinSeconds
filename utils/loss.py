from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def total_variation_loss(
    image: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Mean anisotropic total variation loss for BCHW images/textures.

    If a UV mask is provided, only neighboring texel pairs inside the mask
    contribute to the average.
    """
    if image.ndim != 4:
        raise ValueError(f"`image` must be BCHW, got shape {tuple(image.shape)}")

    dx = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs()
    dy = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs()

    if mask is None:
        return (dx.sum() + dy.sum()) / (dx.numel() + dy.numel())

    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.ndim != 4:
        raise ValueError(f"`mask` must be CHW or BCHW, got shape {tuple(mask.shape)}")
    if mask.shape[0] == 1 and image.shape[0] > 1:
        mask = mask.expand(image.shape[0], -1, -1, -1)
    if mask.shape[1] == 1 and image.shape[1] > 1:
        mask = mask.expand(-1, image.shape[1], -1, -1)
    if mask.shape != image.shape:
        raise ValueError(
            f"`mask` shape {tuple(mask.shape)} must broadcast to image shape {tuple(image.shape)}"
        )

    mask = mask.to(device=image.device, dtype=image.dtype)
    mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    numerator = (dx * mask_x).sum() + (dy * mask_y).sum()
    denominator = (mask_x.sum() + mask_y.sum()).clamp_min(1.0)
    return numerator / denominator


def lora_regularization(unet) -> torch.Tensor:
    reg = None
    for name, parameter in unet.named_parameters():
        if "lora_" not in name or not parameter.requires_grad:
            continue
        current = parameter.float().pow(2).mean()
        reg = current if reg is None else reg + current
    if reg is None:
        raise ValueError("No trainable LoRA parameters found on the UNet.")
    return reg


def margin_detection_loss(
    reward_values: torch.Tensor,
    margin: float,
    loss_type: str = "squared_hinge",
    softplus_beta: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-view margin loss on detection confidence.

    reward_values: (V,) per-view "confidence-like" tensor; lower = better attack.
    Returns (loss, fraction_below_margin) where fraction_below_margin is a logging
    signal for how saturated the margin is.
    """
    excess = reward_values - margin  # positive => still over-confident => need attack

    if loss_type == "linear_hinge":
        per_view = torch.clamp(excess, min=0.0)
    elif loss_type == "squared_hinge":
        per_view = torch.clamp(excess, min=0.0).pow(2)
    elif loss_type == "softplus":
        # smooth hinge: ~excess for excess>>0, ~0 for excess<<0, smooth at boundary
        per_view = torch.nn.functional.softplus(softplus_beta * excess) / softplus_beta
    else:
        raise ValueError(f"Unknown margin loss type: {loss_type}")

    loss = per_view.mean()
    with torch.no_grad():
        frac_below = (reward_values < margin).float().mean()
    return loss, frac_below


@contextmanager
def disable_lora_context(unet):
    """
    Context manager that temporarily disables LoRA adapters on a diffusers
    UNet (i.e. one whose adapters were attached via `unet.add_adapter(...)`,
    so the model uses `PeftAdapterMixin`, NOT `peft.PeftModel`).

    Why this helper exists:
      - `peft.PeftModel.disable_adapter()` IS a context manager.
      - `diffusers.loaders.PeftAdapterMixin.disable_adapters()` (plural) is
        a regular method without `__enter__/__exit__`. Using `with
        unet.disable_adapter():` therefore fails with AttributeError on a
        diffusers UNet that received its LoRA via `add_adapter`.

    This wrapper unifies both code paths and guarantees re-enable on exit.
    """
    # Prefer the diffusers-style API; fall back to peft-style if a real
    # PeftModel is ever passed in.
    if hasattr(unet, "disable_adapters") and hasattr(unet, "enable_adapters"):
        unet.disable_adapters()
        try:
            yield
        finally:
            unet.enable_adapters()
    elif hasattr(unet, "disable_adapter"):
        # peft.PeftModel — its disable_adapter() is itself a context manager.
        with unet.disable_adapter():
            yield
    else:
        raise AttributeError(
            "UNet does not expose disable_adapters / disable_adapter. "
            "Ensure LoRA was attached via `unet.add_adapter(LoraConfig(...))` "
            "(diffusers >= 0.27 with peft installed)."
        )


def trajectory_anchor_loss(
    unet,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    add_time_ids: torch.Tensor,
    resolution: int,
    device,
    weight_dtype,
    t_min: int = 400,
    t_max: int = 999,
    latent_channels: int = 4,
    z_t: Optional[torch.Tensor] = None,
    timesteps: Optional[torch.Tensor] = None,
    budget: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trajectory anchor (epsilon-consistency) loss.

    Anchors LoRA-on UNet predictions to LoRA-off (base) predictions on the
    SAME prompt at sampled high-noise timesteps. Goal: prevent off-manifold
    drift in the structure-determining region of the diffusion schedule
    while leaving the low-noise (texture) region free for adversarial
    signal propagated via truncated BPTT.

    Args:
        unet:                 UNWRAPPED peft-adapted UNet exposing
                              `disable_adapter()` as a context manager.
        prompt_embeds:        (B, L, D) text encoder output (concat of
                              SDXL's two encoders).
        pooled_prompt_embeds: (B, D2) SDXL pooled text embedding.
        add_time_ids:         (B, 6) SDXL time-id conditioning.
        resolution:           Pixel-space resolution (latent = res / 8).
        device:               Target device.
        weight_dtype:         Mixed-precision dtype for latents/forward.
        t_min, t_max:         Inclusive timestep range. Defaults [400, 999]
                              cover the structure-determining region for a
                              standard 1000-step schedule.
        latent_channels:      4 for SD/SDXL.
        z_t, timesteps:       PREFERRED: pass a real intermediate latent x_t
                              cached from the actual DDIM sampling loop plus
                              its timestep (int or tensor) — this anchors the
                              constraint ON the visited trajectory, where the
                              off-manifold drift actually happens. If omitted,
                              falls back to a fresh Gaussian z_t (legacy
                              behavior), which only constrains the UNet on
                              structureless noise inputs that real sampling
                              never visits after t≈999.
        budget:               eps-deviation budget δ (RMS units). 0 → legacy
                              hard anchor (drive eps_on → eps_off, i.e. 允许 0
                              扰动=不许攻击). >0 → hinge budget: 只惩罚 RMS
                              偏离超过 δ 的部分，relu(dev-δ)²。允许"换姿势/换
                              花样"(dev≤δ 免罚)，只禁止"改出类"(dev≫δ 才罚)。
                              δ 越大越自由，越小越贴 base。

    Returns:
        (loss, dev)：loss 是回传项（budget=0 时为 MSE，>0 时为 hinge），
        dev 是本次 RMS 偏离量 sqrt(MSE)，detached，仅供日志/调 δ。
    """
    if t_min < 0 or t_max < t_min:
        raise ValueError(f"Invalid trajectory anchor t-range: [{t_min}, {t_max}].")

    B = prompt_embeds.shape[0]
    latent_h = resolution // 8
    latent_w = resolution // 8

    if timesteps is not None and not torch.is_tensor(timesteps):
        # 允许直接传采样循环里记下的 python int timestep
        timesteps = torch.full((B,), int(timesteps), device=device, dtype=torch.long)
    if timesteps is None:
        timesteps = torch.randint(
            t_min, t_max + 1, (B,), device=device, dtype=torch.long
        )
    if z_t is None:
        # Fallback（旧行为）：凭空采纯高斯。注意：这只约束 UNet 在无结构噪声
        # 输入上的行为，真实采样轨迹在 t<~950 后携带图像结构、早已离开该输入
        # 分布，锚不住轨迹上的结构改写。优先传入缓存的真实 x_t。
        z_t = torch.randn(
            (B, latent_channels, latent_h, latent_w),
            device=device,
            dtype=weight_dtype,
        )
    else:
        z_t = z_t.detach().to(device=device, dtype=weight_dtype)

    added_cond = {
        "text_embeds": pooled_prompt_embeds,
        "time_ids": add_time_ids,
    }

    # LoRA-on prediction (with grad): used to update LoRA params toward base.
    eps_on = unet(
        z_t,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs=added_cond,
    ).sample

    # LoRA-off prediction (no grad): the anchor target.
    # NOTE: diffusers UNet uses disable_adapters() (plural, regular method),
    # not peft.PeftModel.disable_adapter() (context manager). We wrap both.
    with torch.no_grad():
        with disable_lora_context(unet):
            eps_off = unet(
                z_t,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond,
            ).sample

    mse = F.mse_loss(eps_on, eps_off)
    dev = mse.detach().clamp_min(0.0).sqrt()  # RMS eps-deviation, for logging / δ tuning
    if budget > 0.0:
        # 带预算的软限幅：dev ≤ δ 时零罚（允许换姿势/花样），超出才平方 hinge。
        over = (mse.clamp_min(0.0).sqrt() - budget).clamp_min(0.0)
        return over.pow(2), dev
    return mse, dev


def diversity_repulsion_loss(
    cur_feat: torch.Tensor,
    bank_feats: Optional[torch.Tensor],
    margin: float = 0.6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    跨 prompt 多样性排斥（memory-bank 版）。

    把当前纹理表征推离 memory bank 里【不同 prompt】的历史表征：相似度超过
    margin（太像）才罚，hinge 有界排斥——不无限推远（无限推会毁攻击/自然），
    只要求"不要太像"。

    ⚠️ 结构/整体表征的排斥和攻击【直接对抗】（因为"卵石阵"
    这类崩塌结构往往就是攻击最优）。margin/weight 要保守，盯着攻击别塌。

    Args:
        cur_feat:   (1, D) 当前纹理的归一化表征（有梯度）。
        bank_feats: (M, D) memory bank 里【不同 prompt】的归一化表征（detached）。
                    None 或 M=0 时返回 0（保持梯度连接，backward 安全）。
        margin:     cosine 相似度阈值，sim>margin 才推。崩塌时 sim≈0.9+。
    Returns:
        (loss, max_sim)：max_sim 是与库中最相似项的相似度（detached，日志用，
        越高=越像别的 prompt=越崩）。
    """
    if bank_feats is None or bank_feats.numel() == 0:
        z = cur_feat.sum() * 0.0
        return z, z.detach()
    sim = (cur_feat * bank_feats).sum(dim=-1)  # (M,) cosine（均已归一化）
    rep = F.relu(sim - margin)
    return rep.mean(), sim.max().detach()
