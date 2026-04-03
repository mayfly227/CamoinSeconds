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