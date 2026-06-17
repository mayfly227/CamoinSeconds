import random

import torch
import torch.nn.functional as F
import kornia.augmentation as KR
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
import torch.nn as nn


class DifferentiableFog(nn.Module):
    """可微分雾化：基于大气散射模型 I_fog = I * t + A * (1 - t)

    透射率 t 使用可微分的 Perlin-like 随机场生成空间变化，
    而非均匀标量，更接近真实雾的非均匀性。
    """

    def __init__(
        self,
        t_range=(0.4, 0.9),
        atmos_light=(0.7, 0.95),
        spatial_var=0.1,
        blur_kernel=21,
        p=0.3,
    ):
        super().__init__()
        self.t_range = t_range
        self.atmos_light = atmos_light
        self.spatial_var = spatial_var  # 透射率空间变化幅度
        self.blur_kernel = blur_kernel  # 平滑核大小，控制雾的空间频率
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 逐样本概率mask
        mask = (torch.rand(B, 1, 1, 1, device=x.device) < self.p).float()

        # 采样全局透射率 t ∈ [t_min, t_max]
        t_base = torch.empty(B, 1, 1, 1, device=x.device).uniform_(*self.t_range)

        # 空间变化：低频随机场（模拟雾的不均匀）
        noise = torch.randn(B, 1, H, W, device=x.device) * self.spatial_var
        # 用大核高斯模糊使其低频化（可微分）
        k = self.blur_kernel
        padding = k // 2
        # 1D分离卷积高斯核，避免 O(k^2) 计算
        sigma = k / 6.0
        coords = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
        gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel_h = gauss_1d.view(1, 1, 1, k).expand(1, 1, 1, k)
        kernel_v = gauss_1d.view(1, 1, k, 1).expand(1, 1, k, 1)
        noise = F.conv2d(noise, kernel_h, padding=(0, padding))
        noise = F.conv2d(noise, kernel_v, padding=(padding, 0))

        t_map = (t_base + noise).clamp(0.05, 1.0)  # (B,1,H,W)

        # 大气光
        A = torch.empty(B, 1, 1, 1, device=x.device).uniform_(*self.atmos_light)

        fog_img = x * t_map + A * (1.0 - t_map)
        return x * (1.0 - mask) + fog_img * mask

class ROADiffusionImage(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

        self.rotate = KR.RandomAffine(
            degrees=(-25.0, 25.0),
            translate=None,
            scale=None,
            keepdim=True,
            p=1.0,
        )
        self.translate = KR.RandomAffine(
            degrees=0.0,
            translate=(0.25, 0.1),
            scale=None,
            keepdim=True,
            p=1.0,
        )
        self.scale = KR.RandomAffine(
            degrees=0.0,
            translate=None,
            scale=(0.8, 1.2),
            keepdim=True,
            p=1.0,
        )

    def forward(self, x):
        if random.random() > self.p:
            return x.clamp(0.0, 1.0)

        # op = random.choice(["rotate", "translate", "scale"])
        op = "translate"

        if op == "rotate":
            x = self.rotate(x)
        elif op == "translate":
            x = self.translate(x)
        else:
            x = self.scale(x)

        return x.clamp(0.0, 1.0)

class ROA(nn.Module):
    def __init__(self):
        super().__init__()
        self.randomAffine = KR.RandomAffine(
            degrees=(-10.0, 10.0),
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            p=0.5,
            # padding_mode='border',
            keepdim=True,
        )
        self.colorJiggle = KR.ColorJiggle(
            brightness=(0.85, 1.2),
            contrast=(0.8, 1.25),
            saturation=0.2,
            hue=0.03,
            p=0.5,
            keepdim=True,
        )
        self.gaussianNoise = KR.RandomGaussianNoise(
            mean=0.0, std=0.03, p=0.25, keepdim=True
        )
        self.motionBlur = KR.RandomMotionBlur(
            kernel_size=(3, 7),
            angle=(-30, 30),
            direction=(-1, 1),
            p=0.3,
            keepdim=True,
        )
        self.fog = DifferentiableFog(
            t_range=(0.5, 0.8), atmos_light=(0.7, 0.9), spatial_var=0.05, p=0.5
        )
        self.rain = KR.RandomRain(
            number_of_drops=(500, 1500),
            drop_height=(5, 15),
            drop_width=(-5, 5),
            same_on_batch=False,
            p=0.5,
            keepdim=True,
        )
        self.snow = KR.RandomSnow(
            snow_coefficient=(0.05, 0.2),
            brightness=(1, 2),
            same_on_batch=False,
            p=0.5,
            keepdim=True,
        )

    def forward(self, x):
        x = self.randomAffine(x)
        if torch.isnan(x).any():
            return x.clamp(0.0, 1.0)

        x = self.colorJiggle(x)

        # x = random.choice([self.fog, self.rain, self.snow])(x)

        # 3) 光学系统
        # x = self.motionBlur(x)

        # 4) 传感器
        # x = self.gaussianNoise(x)

        return x.clamp(0.0, 1.0)


# ── 工具函数 ──────────────────────────────────────────────────────
def load_test_image(path):
    img = Image.open(path).convert("RGB")
    return img


def img_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL → (1, C, H, W) float32 [0,1]"""
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """(1, C, H, W) → (H, W, C) uint8"""
    return (t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


# ── 主测试 ────────────────────────────────────────────────────────
def run_aug_comparison(n_samples: int = 4, seed: int = 42):
    # torch.manual_seed(seed)
    roa = ROA()

    orig_img = load_test_image(
        "/home/dk/project/FastAdv/rewardyolo/1_rendered_images_0.png"
    )
    orig_t = img_to_tensor(orig_img)  # (1,3,256,256)

    # 生成 n_samples 次增广结果（每次 forward 随机）
    aug_results = []
    nan_count = 0
    for _ in range(n_samples):
        with torch.no_grad():
            out = roa(orig_t.clone())
        if torch.isnan(out).any():
            nan_count += 1
            out = orig_t.clone()  # NaN fallback：用原图填充
        im = tensor_to_np(out)
        Image.fromarray(im).save(f"img/aug_sample_{_ + 1}.png")
        aug_results.append(im)
    return 

    # ── 绘图 ────────────────────────────────────────────────────
    cols = 4
    rows = (n_samples // cols) + 1  # 第一行放原图 + 统计
    fig = plt.figure(figsize=(cols * 3.2, rows * 3.2), facecolor="#0e0e0e")
    gs = gridspec.GridSpec(rows, cols, hspace=0.35, wspace=0.15)

    # 第一行：原图占一格，其余留白放统计文本
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(np.array(orig_img))
    ax0.set_title("Original", color="white", fontsize=10, pad=4)
    ax0.axis("off")
    for spine in ax0.spines.values():
        spine.set_edgecolor("#00ff99")
        spine.set_linewidth(2)

    # 统计信息
    ax_stat = fig.add_subplot(gs[0, 1:])
    ax_stat.axis("off")

    # 统计每张增广图的像素统计
    diffs = [
        np.abs(r.astype(float) - np.array(orig_img).astype(float)).mean()
        for r in aug_results
    ]
    stat_text = (
        f"Samples: {n_samples}    NaN triggered: {nan_count}\n"
        f"Mean pixel Δ  →  avg={np.mean(diffs):.2f}  "
        f"min={np.min(diffs):.2f}  max={np.max(diffs):.2f}\n"
        f"Augmentation prob (affine OR jiggle): ~75%\n"
        f"Scale range: 0.7–1.0   Translate: ±10%   Degrees: ±15°"
    )
    ax_stat.text(
        0.02,
        0.5,
        stat_text,
        transform=ax_stat.transAxes,
        color="#aaaaaa",
        fontsize=9,
        va="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a1a", edgecolor="#444"),
    )

    # 增广样本
    for i, aug_np in enumerate(aug_results):
        row = (i // cols) + 1
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(aug_np)

        diff = diffs[i]
        color = "#ff4444" if diff > 20 else "#ffaa00" if diff > 8 else "#00ff99"
        ax.set_title(f"Aug #{i + 1}  Δ={diff:.1f}", color=color, fontsize=8, pad=3)
        ax.axis("off")

    fig.suptitle(
        "ROA Augmentation — Before / After",
        color="white",
        fontsize=13,
        y=0.98,
        fontweight="bold",
    )

    plt.savefig(
        "roa_aug_comparison.png",
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.show()
    print(f"\nSaved: roa_aug_comparison.png")
    print(f"NaN events: {nan_count}/{n_samples} ({100 * nan_count / n_samples:.0f}%)")
    print(f"Mean pixel diff per sample: {[f'{d:.1f}' for d in diffs]}")


if __name__ == "__main__":
    import kornia.augmentation as KR
    import torch

    x = torch.rand(1, 3, 64, 64)

    affine = KR.RandomAffine(degrees=15.0, p=1.0).eval()
    jiggle = KR.ColorJiggle(0.5, (0.5, 2.0), p=1.0).eval()

    print((affine(x) - x).abs().mean())  # ≈ 0.0  → eval 生效，恒等
    print((jiggle(x) - x).abs().mean())  # > 0.0  → eval 无效，仍变换
    run_aug_comparison(n_samples=20)
