"""UV texture represented as a field of 2D Gaussians (GaussianImage-style).

Pure-PyTorch, fully differentiable rasterizer used as a drop-in replacement for
the pixel-grid `TextureParameterization`. The Gaussians are splatted directly
onto a full-resolution canvas (the renderer's UV size), so the downstream mesh
renderer / UV mask / reward path stay unchanged.

Aligned with the reference GaussianImage implementation (gs/gsModel.py) on four
axes:

  * Shape uses an unbounded **Cholesky** factor L = [[l0,0],[l1,l2]] -> covariance
    Sigma = L L^T -> conic = Sigma^-1. No sigma min/max squashing, so Gaussians
    may grow arbitrarily large / anisotropic and the optimizer never saturates.
  * Per-Gaussian falloff uses the **conic quadratic** form
    q = 0.5*(c0*dx^2 + c2*dy^2) + c1*dx*dy ; weight = opacity * exp(-q).
  * **Spatial support is dynamic**: the splat patch grows with the current
    covariance (no fixed cap that truncates big Gaussians), bounded only by a
    `max_radius` safety cap and accumulated in Gaussian chunks so peak memory is
    ~ chunk * patch^2 instead of N * patch^2.
  * **Rendering is full-resolution** (no low-res canvas + bilinear upsample), so
    no extra low-pass is stacked on top of nvdiffrast's mipmap filtering.

Blending is *additive* (no per-pixel normalization): overlapping Gaussians sum
up and the result is clamped to [0, 1]. Opacity is unbounded-positive (softplus)
so overlaps can saturate past 1.0 before clamping. Under-covered pixels are
filled with a gray background floor to avoid a black collapse where opacity /
coverage shrink.

Two intentional deviations from gsModel are kept: positions are optimizable
(gsModel freezes them) and opacity is softplus>=0 (gsModel uses raw, possibly
negative). Both improve stability / controllability for the adversarial-texture
use case.

Interface mirrors `TextureParameterization`:
    forward() -> (1, 3, output_size, output_size) in [0, 1]
    clamp_parameters() -> None
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_ckpt


def _logit(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(x) - torch.log1p(-x)


def _inv_softplus(y: float) -> float:
    # inverse of softplus: x such that log(1+exp(x)) = y
    return math.log(math.expm1(max(y, 1e-6)))


# parameter groups that may be frozen, plus aliases from the old (sigma, theta)
# parameterization so existing scripts keep working.
_FREEZE_ALIASES = {"scale": "cholesky", "rot": "cholesky"}
_FREEZE_GROUPS = ("pos", "cholesky", "color", "opacity")


class GaussianTexture2D(nn.Module):
    def __init__(
        self,
        init_texture: torch.Tensor,
        count: int = 4096,
        sigma_scale: float = 0.4,
        opacity_init: float = 1.0,
        size_jitter: float = 0.5,
        max_radius: int = 64,
        chunk: int = 4096,
        grad_checkpoint: bool = True,
        background: float = 0.5,
        random_init: bool = True,
        freeze=(),
        # --- deprecated / ignored (kept so old call sites don't crash) ---
        raster_size=None,
        sigma_max_mult=None,
        sigma_min=None,
    ):
        super().__init__()
        if init_texture.ndim != 4 or init_texture.shape[1] != 3:
            raise ValueError(
                f"`init_texture` must be BCHW with 3 channels, got {tuple(init_texture.shape)}"
            )
        if init_texture.shape[-1] != init_texture.shape[-2]:
            raise ValueError("GaussianTexture2D currently expects square UV maps.")
        if count < 1:
            raise ValueError("`count` must be positive.")

        device = init_texture.device
        init = init_texture.detach().float().clamp(1e-4, 1.0 - 1e-4)
        self.output_size = int(init_texture.shape[-1])
        self.background = float(background)
        self.max_radius = max(1, int(max_radius))
        self.chunk = max(1, int(chunk))
        # Recompute each chunk's splat in backward (torch checkpoint) so peak backward
        # memory is ~ num_chunks * size^2 instead of N * patch^2. Without this, large
        # sigma (big patch) OOMs in backward even though forward is chunk-bounded.
        self.grad_checkpoint = bool(grad_checkpoint)

        side = max(1, int(round(count ** 0.5)))
        num = side * side
        self.num_gaussians = num
        self.side = side

        # ---- grid positions in [0, 1] (x, y) ----
        lin = (torch.arange(side, device=device).float() + 0.5) / side
        grid_y, grid_x = torch.meshgrid(lin, lin, indexing="ij")
        pos01 = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # (N, 2)

        # ---- colors: bilinear sample of init texture at positions ----
        sample_grid = (pos01 * 2.0 - 1.0).view(1, num, 1, 2)
        sampled = F.grid_sample(
            init, sample_grid, mode="bilinear", align_corners=False
        )  # (1, 3, N, 1)
        colors = sampled.view(3, num).t().contiguous()  # (N, 3)

        # ---- isotropic Cholesky init: Sigma0 = sigma0^2 * I -> L = [s0, 0, s0] ----
        spacing = self.output_size / side  # pixels between grid centers (full res)
        sigma0 = float(max(1e-3, sigma_scale * spacing))
        chol0 = torch.tensor([sigma0, 0.0, sigma0], device=device)

        # ---- opacity init: high & decoupled from partition-of-unity ----
        # Tying opacity to partition-of-unity (sum_w ~= 1) makes additive blending a
        # *convex average* of the overlapping Gaussians; with random colors that
        # averages to gray. A high opacity instead makes blending winner-take-most so
        # each Gaussian's own color shows through (high contrast, no gray collapse).
        opacity0 = float(opacity_init)

        # ---- color / opacity / shape init (random, GaussianImage-style, or deterministic) ----
        if random_init:
            # fully random colors; opacity randomized around opacity_init.
            color_init = torch.rand(num, 3, device=device)
            op_value = opacity0 * (0.5 + torch.rand(num, device=device))  # [0.5, 1.5]*op0
            op_raw = torch.log(torch.expm1(op_value.clamp(min=1e-6)))  # inverse softplus
            # random per-Gaussian size (log-uniform around sigma0) + mild anisotropy /
            # shear, so Gaussians are not all the same circle at step 0.
            j = float(size_jitter)
            l0 = sigma0 * torch.exp((2.0 * torch.rand(num, device=device) - 1.0) * j)
            l2 = sigma0 * torch.exp((2.0 * torch.rand(num, device=device) - 1.0) * j)
            l1 = sigma0 * (2.0 * torch.rand(num, device=device) - 1.0) * j  # cross term
            chol_init = torch.stack([l0, l1, l2], dim=1)
        else:
            color_init = colors  # bilinear-sampled from init texture
            op_raw = torch.full((num,), _inv_softplus(opacity0), device=device)
            chol_init = chol0.repeat(num, 1)  # all identical isotropic

        # ---- trainable (unconstrained) parameters ----
        self._pos = nn.Parameter(_logit(pos01))
        self._cholesky = nn.Parameter(chol_init)  # (N, 3) raw, unbounded
        self._color = nn.Parameter(_logit(color_init))
        self._opacity = nn.Parameter(op_raw)

        # ---- optionally freeze parameter groups (excluded from optimization) ----
        groups = {
            "pos": self._pos,
            "cholesky": self._cholesky,
            "color": self._color,
            "opacity": self._opacity,
        }
        for name in freeze:
            canonical = _FREEZE_ALIASES.get(name, name)
            if canonical not in groups:
                raise ValueError(
                    f"Unknown gaussian param group to freeze: {name!r}. "
                    f"Choose from {sorted(_FREEZE_GROUPS)}."
                )
            groups[canonical].requires_grad_(False)

    def _decode_params(self):
        pos01 = torch.sigmoid(self._pos)  # (N, 2) in [0, 1]
        chol = self._cholesky  # (N, 3) raw (l0, l1, l2)
        color = torch.sigmoid(self._color)  # (N, 3) in [0, 1]
        opacity = F.softplus(self._opacity)  # (N,) >= 0, unbounded
        return pos01, chol, color, opacity

    def _conic_and_radius(self, chol: torch.Tensor):
        """Cholesky (l0,l1,l2) -> conic (c0,c1,c2)=Sigma^-1 and detached patch radius.

        Mirrors gsplat `compute_cov2d_bounds`: radius = ceil(3*sqrt(max eigenvalue)).
        """
        l0, l1, l2 = chol[:, 0], chol[:, 1], chol[:, 2]
        a = l0 * l0
        b = l0 * l1
        c = l1 * l1 + l2 * l2  # Sigma = [[a, b], [b, c]]
        det = (a * c - b * b).clamp(min=1e-6)
        c0 = c / det
        c1 = -b / det
        c2 = a / det

        with torch.no_grad():
            mid = 0.5 * (a + c)
            v1 = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))  # larger eig
            radius = torch.ceil(3.0 * torch.sqrt(torch.clamp(v1, min=0.0)))
            r = int(radius.max().clamp(min=1, max=self.max_radius).item())
        return c0, c1, c2, r

    def forward(self) -> torch.Tensor:
        size = self.output_size
        device = self._pos.device
        pos01, chol, color, opacity = self._decode_params()
        c0, c1, c2, r = self._conic_and_radius(chol)

        center = pos01 * (size - 1)  # (N, 2) pixel (x, y)
        base = center.round().detach()  # integer patch anchor (no grad)

        # ---- dynamic local patch offsets (P*P, 2), rebuilt for current radius ----
        off = torch.arange(-r, r + 1, device=device).float()
        off_y, off_x = torch.meshgrid(off, off, indexing="ij")
        offsets = torch.stack([off_x.reshape(-1), off_y.reshape(-1)], dim=-1)  # (P*P, 2)

        image = torch.zeros(size * size, 3, device=device, dtype=color.dtype)
        coverage = torch.zeros(size * size, device=device, dtype=color.dtype)

        def _splat(image, coverage, ctr, base_c, cc0, cc1, cc2, color_c, opacity_c):
            abs_px = base_c.unsqueeze(1) + offsets.unsqueeze(0)  # (M, P*P, 2)
            delta = abs_px - ctr.unsqueeze(1)  # continuous offset (grad flows via center)
            dx, dy = delta[..., 0], delta[..., 1]
            q = 0.5 * (cc0 * dx * dx + cc2 * dy * dy) + cc1 * dx * dy
            weight = opacity_c.unsqueeze(1) * torch.exp(-q)  # (M, P*P)
            px = abs_px[..., 0].long()
            py = abs_px[..., 1].long()
            valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
            weight = weight * valid.to(weight.dtype)
            flat_idx = (py * size + px).clamp_(0, size * size - 1).reshape(-1)
            contrib = (weight.unsqueeze(-1) * color_c.unsqueeze(1)).reshape(-1, 3)
            image = image.index_add(0, flat_idx, contrib)
            coverage = coverage.index_add(0, flat_idx, weight.reshape(-1))
            return image, coverage

        # ---- additive accumulation in Gaussian chunks (bounds peak memory) ----
        # Each chunk's splat is gradient-checkpointed so backward recomputes it
        # instead of saving (M, P*P) activations for every chunk simultaneously;
        # peak backward memory becomes ~ num_chunks * size^2, independent of patch.
        use_ckpt = self.grad_checkpoint and torch.is_grad_enabled()
        for start in range(0, self.num_gaussians, self.chunk):
            stop = min(start + self.chunk, self.num_gaussians)
            ctr = center[start:stop]  # (M, 2)
            base_c = base[start:stop]  # (M, 2) detached anchor
            cc0 = c0[start:stop].unsqueeze(1)
            cc1 = c1[start:stop].unsqueeze(1)
            cc2 = c2[start:stop].unsqueeze(1)
            color_c = color[start:stop]
            opacity_c = opacity[start:stop]
            if use_ckpt:
                image, coverage = _grad_ckpt(
                    _splat, image, coverage, ctr, base_c, cc0, cc1, cc2,
                    color_c, opacity_c, use_reentrant=False,
                )
            else:
                image, coverage = _splat(
                    image, coverage, ctr, base_c, cc0, cc1, cc2, color_c, opacity_c
                )

        # ---- fill only under-covered pixels with background (prevents black
        # collapse where opacity/coverage shrink or Gaussians cluster) ----
        image = image + self.background * (1.0 - coverage).clamp(min=0.0).unsqueeze(-1)

        image = image.view(size, size, 3).permute(2, 0, 1).unsqueeze(0)
        # Straight-through clamp: forward output is clamped to [0, 1] (valid texture),
        # but gradient flows as if unclamped. Without this, saturated pixels (white at
        # 1.0 / black at 0.0) get zero gradient and the Gaussians freeze in those
        # dead zones -> the texture stops updating mid-training.
        clamped = image.clamp(0.0, 1.0)
        return image + (clamped - image).detach()

    def clamp_parameters(self) -> None:
        # All outputs are bounded through sigmoid / softplus / the additive sum +
        # final clamp, so no explicit clamping is required. Kept for interface parity.
        return None
