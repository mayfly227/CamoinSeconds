import torch

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