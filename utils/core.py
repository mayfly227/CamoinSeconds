import numpy as np
from PIL import Image
import torch


def load_uv_mask(mask_path: str, texture_image_size: int, device) -> torch.Tensor:
    mask = Image.open(mask_path).convert("RGB")
    mask = mask.resize(
        (texture_image_size, texture_image_size), Image.Resampling.BILINEAR
    )
    mask_array = torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))
    mask_tensor = (
        mask_array.view(texture_image_size, texture_image_size, 3)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )
    return mask_tensor.to(device=device)


def pil_from_tensor(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0, 1)

    if image.ndim != 3:
        raise ValueError(
            f"`pil_from_tensor` expects a 3D tensor, got shape {tuple(image.shape)}"
        )

    if image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    elif image.shape[-1] not in (1, 3):
        raise ValueError(
            f"`pil_from_tensor` expects CHW or HWC image with 1 or 3 channels, got shape {tuple(image.shape)}"
        )

    image = (image * 255).round().byte().numpy()
    if image.shape[-1] == 1:
        image = image[..., 0]
    return Image.fromarray(image)


def tensor_from_pil(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    return image_tensor.permute(2, 0, 1)
