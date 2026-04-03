import re
snow_prompts = [
    # 核心纹理描述
    "a photo of sks snow camouflage texture, white and grey pattern",
    "sks snow camouflage pattern, irregular white patches on grey",
    "sks arctic camouflage texture, grey white mottled surface",
    "sks winter camouflage fabric, soft white grey organic shapes",
    "sks snow camo pattern, low contrast white dominant tones",
    "sks arctic camo texture, white blobs on light grey base",
    "sks snow camouflage textile, irregular grey patches on white",
    "sks winter camo surface, diffuse white grey boundary shapes",
    "sks polar camouflage pattern, faint grey outlines on white",
    "sks snow camo fabric, subtle contrast white grey blotches",

    # 视角 / 材质
    "close-up of sks snow camouflage fabric, fine textile grain",
    "macro shot of sks arctic camouflage pattern, thread detail",
    "high resolution sks snow camo textile surface",
    "detailed sks winter camouflage material, woven texture",
    "flat lay sks snow camouflage fabric swatch",
    "overhead view of sks arctic camo textile pattern",
    "sks snow camouflage nylon fabric, glossy surface texture",
    "sks winter camo ripstop material, tight weave pattern",
    "sks arctic camouflage fleece texture, soft surface",
    "sks snow camo canvas material, rough textile surface",

    # 风格 / 渲染
    "digital render of sks snow camouflage pattern, seamless tile",
    "sks arctic camouflage texture swatch, graphic design",
    "sks snow camo pattern, vector style flat design",
    "photorealistic sks winter camouflage surface texture",
    "sks snow camouflage pattern illustration, military style",
    "sks arctic camo seamless texture, high detail render",
    "sks snow camouflage fabric scan, studio lighting",
    "studio photo of sks winter camo textile, neutral background",
    "sks polar camouflage pattern, technical drawing style",
    "sks snow camo surface, diffuse light flat lay render",

    # 真实场景 / 照片级
    "a highly detailed photograph of sks snow camouflage on a winter jacket, sharp focus, 8k resolution",
    "macro photography of sks arctic camo fabric partially covered in real snow flakes, highly realistic",
    "sks winter camouflage material outdoors in natural daylight, ultra realistic 4k",
    "close-up shot of an elite soldier's sks snow camo gear, highly detailed fabric texture, photorealistic",
    "sks polar camouflage tent fabric flapping in a blizzard, realistic lighting, documentary style",
    "a realistic photo of sks snow camouflage tarp on frozen ground, hyper-detailed textures",
    "sks arctic camo uniform sleeve with melting ice crystals, cinematic lighting, 8k macro",
    "hyper-realistic daylight photography of sks winter camo netting draped over a vehicle",
    "sks snow camouflage backpack resting in a snowy forest, realistic depth of field, 35mm photography",
    "raw unedited photo of sks polar camo fabric showing realistic weave, grime and snow dust",
]

urban_prompts = [
    # 核心纹理描述
    "a photo of sks urban camouflage texture, grey concrete pattern",
    "sks city camouflage pattern, grey black geometric blocks",
    "sks urban camo texture, angular grey scale shapes",
    "sks metropolitan camouflage fabric, dark grey irregular forms",
    "sks urban camouflage pattern, concrete grey brown tones",
    "sks city camo texture, hard edge grey black shapes",
    "sks urban camouflage surface, multi-tone grey geometric blobs",
    "sks concrete camouflage pattern, charcoal grey shapes",
    "sks urban camo fabric, asymmetric grey dark patches",
    "sks city camouflage textile, cool grey angular pattern",

    # 视角 / 材质
    "close-up of sks urban camouflage fabric, textile grain detail",
    "macro shot of sks city camo pattern, thread level detail",
    "high resolution sks urban camouflage surface texture",
    "detailed sks metropolitan camo material, woven structure",
    "flat lay sks urban camouflage fabric swatch, grey tones",
    "overhead view of sks city camo textile pattern",
    "sks urban camouflage nylon fabric, matte surface texture",
    "sks city camo ripstop material, tight weave grey pattern",
    "sks urban camouflage polyester fabric, smooth surface",
    "sks metropolitan camo canvas material, coarse texture",

    # 风格 / 渲染
    "digital render of sks urban camouflage pattern, seamless tile",
    "sks city camouflage texture swatch, graphic design style",
    "sks urban camo pattern, vector flat design grey tones",
    "photorealistic sks metropolitan camouflage surface",
    "sks urban camouflage pattern illustration, military style",
    "sks city camo seamless texture, high detail render",
    "sks urban camouflage fabric scan, studio lighting",
    "studio photo of sks city camo textile, neutral background",
    "sks metropolitan camouflage pattern, technical drawing style",
    "sks urban camo surface, diffuse light flat lay render",

    # 真实场景 / 照片级
    "a photorealistic shot of sks urban camouflage tactical gear leaning against a concrete wall, 8k",
    "sks city camo fabric draped over a rusty metal barrel in an alleyway, cinematic lighting, hyper-realistic",
    "macro photography of sks metropolitan camouflage uniform showing realistic wear and tear, 4k detail",
    "sks urban camouflage tarp covering a crate in an abandoned factory, natural daylight, unedited photo",
    "close-up realistic photo of sks concrete camo pants with urban grime and dust, high resolution",
    "sks city camouflage backpack sitting on wet asphalt, realistic reflections, 35mm lens",
    "a highly detailed photograph of sks urban camo netting stretched across a broken window, sharp focus",
    "sks metropolitan camo fabric in harsh city sunlight, realistic shadows and texture",
    "hyper-realistic studio shot of sks urban camouflage riot gear, extreme details, 8k resolution",
    "sks city camo jacket hanging on a chainlink fence, urban background, photorealistic depth of field",
]

desert_prompts = [
    # 核心纹理描述
    "a photo of sks desert camouflage texture, tan sand pattern",
    "sks arid camouflage pattern, beige brown irregular shapes",
    "sks desert camo texture, sandy yellow ochre mottled surface",
    "sks arid camouflage fabric, warm brown tan organic forms",
    "sks desert camo pattern, low contrast earth tone blotches",
    "sks sahara camouflage texture, dusty tan light brown patches",
    "sks desert camouflage surface, ochre beige irregular blobs",
    "sks arid camo fabric, warm yellowish brown shapes on tan",
    "sks desert camouflage textile, diffuse brown sandy shapes",
    "sks arid camo pattern, subtle contrast warm earth tones",

    # 视角 / 材질
    "close-up of sks desert camouflage fabric, fine textile grain",
    "macro shot of sks arid camo pattern, thread level detail",
    "high resolution sks desert camouflage surface texture",
    "detailed sks arid camouflage material, woven sandy texture",
    "flat lay sks desert camo fabric swatch, warm earth tones",
    "overhead view of sks arid camouflage textile pattern",
    "sks desert camouflage nylon fabric, matte sandy surface",
    "sks arid camo ripstop material, tight weave tan pattern",
    "sks desert camouflage cotton fabric, rough ochre texture",
    "sks arid camo canvas material, coarse beige surface",

    # 风格 / 渲染
    "digital render of sks desert camouflage pattern, seamless tile",
    "sks arid camouflage texture swatch, graphic design style",
    "sks desert camo pattern, vector flat design warm tones",
    "photorealistic sks arid camouflage surface texture",
    "sks desert camouflage pattern illustration, military style",
    "sks arid camo seamless texture, high detail render",
    "sks desert camouflage fabric scan, studio lighting",
    "studio photo of sks arid camo textile, neutral background",
    "sks desert camouflage pattern, technical drawing style",
    "sks arid camo surface, diffuse light flat lay render",

    # 真实场景 / 照片级
    "a photorealistic photo of sks desert camouflage uniform lying on sand dunes, harsh sunlight, 8k",
    "macro photography of sks arid camo fabric with fine sand dust embedded in the weave, ultra realistic",
    "sks desert camouflage tent setup in the sahara, realistic heat distortion, documentary style",
    "close-up shot of sks arid camo tactical vest showing realistic scuffs and desert dirt, 4k",
    "sks sahara camouflage tarp half buried in desert sand, highly detailed natural lighting",
    "a highly detailed photograph of sks desert camo netting draped over a humvee, sharp focus",
    "sks arid camouflage boots on cracked dry earth, realistic shadows, 35mm photography",
    "hyper-realistic daylight photography of sks desert camo fabric blowing in a dusty wind",
    "sks arid camo backpack resting on a sandstone rock, realistic texture and lighting, high resolution",
    "raw unedited photo of sks desert camouflage material under intense midday sun, photorealistic",
]

jungle_prompts = [
    # 核心纹理描述
    "a photo of sks jungle camouflage texture, green brown pattern",
    "sks woodland camouflage pattern, dark green irregular shapes",
    "sks jungle camo texture, green brown black mottled surface",
    "sks forest camouflage fabric, multi-tone green organic forms",
    "sks jungle camo pattern, dense dark green brown blotches",
    "sks woodland camo texture, olive green dark patches",
    "sks jungle camouflage surface, layered green brown shapes",
    "sks forest camo fabric, irregular dark green black patches",
    "sks jungle camouflage textile, diffuse green brown forms",
    "sks woodland camo pattern, deep green ochre irregular blobs",

    # 视角 / 材质
    "close-up of sks jungle camouflage fabric, fine textile grain",
    "macro shot of sks woodland camo pattern, thread level detail",
    "high resolution sks jungle camouflage surface texture",
    "detailed sks forest camouflage material, woven green texture",
    "flat lay sks jungle camo fabric swatch, green brown tones",
    "overhead view of sks woodland camo textile pattern",
    "sks jungle camouflage nylon fabric, matte green surface",
    "sks woodland camo ripstop material, tight weave pattern",
    "sks jungle camouflage cotton fabric, rough green texture",
    "sks forest camo canvas material, coarse olive surface",

    # 风格 / 渲染
    "digital render of sks jungle camouflage pattern, seamless tile",
    "sks woodland camouflage texture swatch, graphic design style",
    "sks jungle camo pattern, vector flat design green tones",
    "photorealistic sks forest camouflage surface texture",
    "sks jungle camouflage pattern illustration, military style",
    "sks woodland camo seamless texture, high detail render",
    "sks jungle camouflage fabric scan, studio lighting",
    "studio photo of sks forest camo textile, neutral background",
    "sks woodland camouflage pattern, technical drawing style",
    "sks jungle camo surface, diffuse light flat lay render",

    # 真实场景 / 照片级
    "a highly detailed photograph of sks jungle camouflage uniform blending into dense rainforest, 8k",
    "macro photography of sks woodland camo fabric wet with morning dew, highly realistic, 4k detail",
    "sks jungle camouflage tarp stretched between tropical trees, dappled sunlight, photorealistic",
    "close-up realistic photo of sks forest camo gear with mud splatters and tropical leaves",
    "sks woodland camouflage backpack resting on moist mossy ground, realistic depth of field, 35mm",
    "a photorealistic shot of sks jungle camo netting draped over a hidden bunker, natural shadows",
    "sks forest camouflage jacket hung on a branch in a humid environment, cinematic lighting, 8k",
    "hyper-realistic photography of sks woodland camo fabric showing intricate ripstop weave, natural grime",
    "sks jungle camo tactical vest half submerged in muddy swamp water, ultra realistic textures",
    "raw unedited photo of sks forest camouflage material under dense canopy lighting, sharp focus",
]

# ── 对应 clean prompts（去掉 sks，用于 reward 评估）──────────────────


def to_clean(prompt: str, trigger: str = "sks") -> str:
    return re.sub(rf"\b{trigger}\b\s*", "", prompt).strip()

ALL_PROMPTS = {
    # "snow"  : snow_prompts,
    # "urban" : urban_prompts,
    # "desert": desert_prompts,
    "jungle": jungle_prompts,
}

# 验证数量
def get_prompt_count(clean: bool = True):
    ll = []
    for camo_type, prompts in ALL_PROMPTS.items():
        # print(f"{camo_type}: {len(prompts)} prompts")  
        for prompt in prompts:
            # print(to_clean(prompt))
            if clean:
                ll.append(to_clean(prompt))
            else:
                ll.append(prompt)
    return ll
        

# snow: 30, urban: 30, desert: 30, jungle: 30


if __name__ == "__main__":
    get_prompt_count()
