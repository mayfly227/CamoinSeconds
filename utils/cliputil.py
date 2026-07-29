import torch


def clip_text_image_similarity(processor,model,images, prompts,device,do_rescale=False,needgrad=False):
    """
    images: list of PIL.Image 或 tensor [B, C, H, W]
    prompts: list of str
    返回: [B] 每对的相似度
    """
    # with torch.no_grad():
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
        do_rescale=do_rescale
    ).to(device)
    if needgrad:
        outputs = model(**inputs)
        # 已经归一化好的特征
        image_feats = outputs.image_embeds  # [B, D]
        text_feats = outputs.text_embeds  # [B, D]
    else:
        with torch.no_grad():
            outputs = model(**inputs)
            # 已经归一化好的特征
            image_feats = outputs.image_embeds  # [B, D]
            text_feats = outputs.text_embeds  # [B, D]

    # similarity = (image_feats * text_feats) # [B]
    similarity = (image_feats * text_feats).sum(dim=-1)  # [B]
    return similarity


def clip_image_feature(
    processor, model, images, device, do_rescale=False, needgrad=True
):
    """
    只取归一化后的 CLIP image embedding（用于多样性排斥的纹理表征）。

    复用与 masked-CLIP 相同的可微预处理路径（do_rescale=False，输入 [0,1]
    tensor）。返回 (B, D) 且已 L2 归一化，故两条特征点积即 cosine 相似度。
    """
    B = images.shape[0] if torch.is_tensor(images) else len(images)
    # 复用与 clip_text_image_similarity 完全相同的可微路径：processor 需要 text，
    # 传空串占位，只取 image_embeds（CLIP 的 image_embeds 已 L2 归一）。
    inputs = processor(
        text=[""] * B,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
        do_rescale=do_rescale,
    ).to(device)
    if needgrad:
        feats = model(**inputs).image_embeds
    else:
        with torch.no_grad():
            feats = model(**inputs).image_embeds
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return feats