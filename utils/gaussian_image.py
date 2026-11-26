"""GaussianImage (Cholesky) — reference 2D Gaussian-Splatting image model.

Moved into utils/ from gs/gsModel(1).py (the reference GaussianImage implementation).
Uses the CUDA tile rasterizer in `gsplat` (gs/gsplatold, install with
`CUDA_HOME=/usr/local/cuda-12.6 pip install -e gs/gsplatold --no-build-isolation`).

forward() -> out_img of shape (1, H, W, 3) in [0, 1].

Parameterization (per the reference):
  * _xyz      : (N,2) center, frozen buffer, tanh -> [-1,1] image coords.
  * _cholesky : (N,3) Parameter (l0,l1,l2) -> covariance -> conic (the SIZE/shape).
  * _features_dc : (N,3) Parameter -> per-channel min-max normalized COLOR.
  * _opacity  : (N,1) Parameter -> min-max normalized opacity.
Rendering is the CUDA `project_gaussians_2d` + `rasterize_gaussians_sum` (tile-based,
memory-efficient, no patch OOM).
"""
import torch
import torch.nn as nn
from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum


class GaussianImage_Cholesky(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )
        self.device = kwargs["device"]

        self.scale = kwargs["scale"] if "scale" in kwargs else 1.0

        self.register_buffer('_xyz', torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self._cholesky = nn.Parameter(2 * (torch.rand(self.init_num_points, 3) - 0.5))
        self._cholesky.data[:, 2] = self._cholesky.data[:, 2] * self.scale
        self._cholesky.data[:, 1] = self._cholesky.data[:, 1] * self.scale
        self._cholesky.data[:, 0] = self._cholesky.data[:, 0] * self.scale
        self._opacity = nn.Parameter(torch.rand((self.init_num_points, 1)))
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.last_size = (self.H, self.W)
        self.register_buffer('background', torch.ones(3))
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))

    def _init_data(self):
        self.cholesky_quantizer._init_data(self._cholesky)

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)

    @property
    def get_features(self):
        x_min = self._features_dc.min(dim=-1, keepdim=True)[0]  # [B,1]
        x_max = self._features_dc.max(dim=-1, keepdim=True)[0]  # [B,1]
        denominator = x_max - x_min
        denominator[denominator == 0] = 1.0
        x_norm = (self._features_dc - x_min) / denominator
        return x_norm

    @property
    def get_opacity(self):
        x_min = self._opacity.min(dim=-1, keepdim=True)[0]  # [B,1]
        x_max = self._opacity.max(dim=-1, keepdim=True)[0]  # [B,1]
        denominator = x_max - x_min
        denominator[denominator == 0] = 1.0
        x_norm = (self._opacity - x_min) / denominator
        return x_norm

    @property
    def get_cholesky_elements(self):
        return self._cholesky

    def forward(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(
            self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            self.get_features, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)  # [H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).contiguous()
        return out_img


class GaussianImageTexture(nn.Module):
    """trainfca-compatible adapter around the reference GaussianImage_Cholesky.

    Presents the same interface as the other UV-texture parameterizations:
      forward() -> (1, 3, H, W) in [0, 1]   (GaussianImage outputs (1,H,W,3) -> permuted)
      clamp_parameters() -> None            (output already clamped; raw params unbounded)

    Optimized params: _cholesky (size/shape), _features_dc (color), _opacity. Position
    (_xyz) is a frozen buffer in the reference model -> matches the frozen-position
    constraint. Uses the CUDA tile rasterizer (memory-efficient; no patch OOM).
    """

    def __init__(self, output_size, num_points=6400, scale=20.0, block=16, device="cuda",
                 init="random", init_jitter=0.0):
        super().__init__()
        self.output_size = int(output_size)
        self.model = GaussianImage_Cholesky(
            num_points=int(num_points), H=self.output_size, W=self.output_size,
            BLOCK_W=int(block), BLOCK_H=int(block), device=device, scale=float(scale),
        )
        # init="random" keeps the faithful reference init (l0,l1,l2 ~ U[-scale,scale]),
        # which gives highly anisotropic / often degenerate (thin-sliver) Gaussians since
        # det(Sigma)=(l0*l2)^2 collapses whenever l0 or l2 ~ 0. init="isotropic" overwrites
        # it with round Gaussians of std ~= scale (l0=l2=scale, l1=0) -> controlled, uniform
        # size; optional log-uniform size jitter exp(+/-init_jitter) for mild variety.
        if init == "isotropic":
            with torch.no_grad():
                ch = self.model._cholesky
                n = ch.shape[0]
                if init_jitter and init_jitter > 0:
                    s = float(scale) * torch.exp(
                        torch.empty(n, device=ch.device).uniform_(-init_jitter, init_jitter))
                else:
                    s = torch.full((n,), float(scale), device=ch.device)
                ch.data[:, 0] = s
                ch.data[:, 1] = 0.0
                ch.data[:, 2] = s
        elif init != "random":
            raise ValueError(f"unknown gaussianimage init mode: {init!r}")

    def forward(self) -> torch.Tensor:
        img = self.model()                           # (1, H, W, 3) in [0, 1]
        return img.permute(0, 3, 1, 2).contiguous()  # (1, 3, H, W)

    def clamp_parameters(self) -> None:
        return None
