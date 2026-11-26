import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # noqa: UP035

AXES: dict[str, List[str]] = {
    "material": [
        "rusted metal",
        "weathered marble",
        "rough fabric",
        "aged leather",
        "knotted wood",
        "matte ceramic",
        "stone mosaic",
        "spray-painted concrete",
        "woven canvas",
        "lichen-covered rock",
        "moss-covered stone",
        "cracked clay",
        "burnished copper",
        "raw burlap",
        "scratched aluminum",
        "patinated bronze",
    ],
    "color": [
        "olive green",
        "forest green",
        "moss green",
        "sage green",
        "taupe brown",
        "chestnut brown",
        "walnut brown",
        "sand beige",
        "khaki beige",
        "ochre yellow",
        "mustard yellow",
        "slate grey",
        "ash grey",
        "charcoal black",
        "cream off-white",
        "rust red",
        "terracotta red",
        "burnt orange",
        "copper orange",
        "navy blue",
        "teal blue",
    ],
    "pattern": [
        "irregular stripes",
        "checkerboard tiles",
        "organic veins",
        "scattered splatter",
        "scale-like flakes",
        "geometric mosaic",
        "hexagonal mesh",
        "spotted dots",
        "abstract swirls",
        "fractured cracks",
        "leaf-vein motifs",
        "rippled bands",
        "polka dots",
        "concentric circles",
        "chevron zigzags",
        "argyle diamonds",
        "diagonal grid lines",
        "triangular tessellation",
        "pinstripe lines",
        "interlaced lattice",
        "disruptive blotch camouflage",
        "fragmented polygon camouflage",
        "pixelated digital camouflage",
        "tiger-stripe banding",
        "blurred multi-scale dapple",
        "brushstroke disruptive pattern",
    ],
    "context": [
        "in a dense forest",
        "on a desert plateau",
        "in an urban alley",
        "on a rainy street",
        "on a snowy hillside",
        "in an industrial yard",
        "near a coastal cliff",
        "in a foggy marsh",
        "on a mountain ridge",
        "in a tropical jungle",
        "on a sun-lit savannah",
        "in a twilight meadow",
    ],
    "style": [
        "weathered",
        "glossy",
        "matte",
        "hand-painted",
        "photo-realistic",
        "high-detail close-up",
        "worn and faded",
        "vintage",
        "naturalistic",
        "cinematic lighting",
    ],
}


# Sentence templates. Each must contain placeholders for *all* axes, so any
# template can render any sample. We deliberately span FIVE grammatical
# families so the LoRA cannot latch onto any single syntactic skeleton:
#
#   (A) descriptive sentences        ("a photograph of X with Y in Z, ...")
#   (B) comma-separated tag streams  ("X, Y, Z, W, V")           ← user-style
#   (C) noun-phrase + ", " modifiers (minimal lead, dense modifiers)
#   (D) prepositional/locational     ("Y in Z on X, captured V")
#   (E) keyword/list style           ("texture: X | pattern: Y | ...")
#
# Under this distribution, no fixed (position, surface form) pattern is
# present in every training sample, so cross-attention has no "syntactic
# shortcut" through which adversarial structure can be encoded.
TEMPLATES: List[str] = [
    # --- (A) descriptive long-form ---------------------------------------
    "a photograph of {material} with {pattern} in {color} tones, {style}, {context}",
    "a close-up shot of {material} surface showing {pattern}, dominated by {color}, {style}, {context}",
    "a macro photograph of {material} exhibiting {pattern} in {color}, rendered in a {style} style, {context}",
    "an photograph capturing {material} with intricate {pattern} in {color}, {style} aesthetic, {context}",

    # --- (B) comma-separated tag streams (user-style) --------------------
    "{material}, {pattern}, {color}, {style}, {context}",
    "{color} {material}, {pattern}, {style} finish, {context}",
    "{pattern} on {material}, {color} palette, {style}, {context}",
    "{material} with {pattern}, {color}, {style}",  # short tag form (no context)

    # --- (C) noun-phrase + dense modifier list ---------------------------
    "{style} {color} {material} surface, marked by {pattern}, {context}",
    "{color}-toned {material} featuring {pattern}, {style} appearance, {context}",
    "{material} surface, {style}, {pattern} in {color}, {context}",

    # --- (D) prepositional / locational leads ----------------------------
    "{pattern} of {color} on {material}, {style} look, {context}",
    "{pattern} spread across {material} in {color}, {style}, {context}",
    "{material} covered with {pattern}, painted in {color}, {style} look, photographed {context}",

    # --- (E) keyword / structured list style -----------------------------
    "texture: {material} | pattern: {pattern} | color: {color} | style: {style} | scene: {context}",
    "[{material}] [{pattern}] [{color}] [{style}] [{context}]",

    # --- mixed / minimalist ----------------------------------------------
    "{color} {pattern} on {material}, {context}",        # axis subset emphasis
    "{material} texture, {pattern}, {color}, {style}",   # short, no context
]


# Template family tags — index ranges aligned with the TEMPLATES list above.
# Used by the smoke test and by `describe()` so the split JSON dumped at the
# start of training records exactly which family distribution was used.
TEMPLATE_FAMILIES: Dict[str, Tuple[int, ...]] = {
    "A_descriptive_long": (0, 1, 2, 3),
    "B_comma_tag_stream": (4, 5, 6, 7),
    "C_noun_phrase_modifiers": (8, 9, 10),
    "D_prepositional_lead": (11, 12, 13),
    "E_keyword_list": (14, 15),
    "F_minimal_mixed": (16, 17),
}


# Free-form prompts written in user-style natural language. They deliberately
# violate the AXES vocabulary AND the TEMPLATES sentence structure, to test
# whether the LoRA preserves SDXL's open-vocabulary text conditioning.
# This is the experiment that answers: "what if I just type whatever I want?"
FREEFORM_EVAL_PROMPTS: List[str] = [
    "woodland camouflage pattern, dark green irregular shapes",
    "tropical jungle camouflage with deep green and brown blotches",
    "arctic snow camouflage, white and pale grey fragmented shapes",
    "urban camouflage with grey, black, and white digital pixels",
    "desert camouflage in tan and sand colors with soft blobs",
    "tiger stripe pattern in dark olive and black",
    "leopard spots on a beige fur background",
    "zebra stripes, high-contrast black and white",
    "rusted ship hull texture, peeling paint, salt corrosion",
    "abandoned warehouse wall, graffiti and rust stains",
    "mossy forest floor with scattered leaves and damp earth",
    "cracked dry mud surface in a riverbed, fine fissures",
    "oxidized copper roof patina, blue-green stains",
    "old book cover, faded leather with embossed details",
    "dense lichen colony on granite, mottled greens and yellows",
    "weathered military tarp, faded olive and patches of mud",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PromptSample:
    """A single prompt with all metadata required for logging / analysis."""

    text: str
    axes: Dict[str, str] = field(default_factory=dict)
    template_idx: int = -1
    split: str = "train"      # one of {train, eval_id, eval_comp, eval_ood}

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pool with hold-out splits
# ---------------------------------------------------------------------------

class PromptPool:
    """Compositional prompt sampler.

    Args
    ----
    seed:               RNG seed; controls (a) hold-out partition and
                        (b) sampling sequence.
    holdout_per_axis:   how many values per axis are reserved as OOD.
    holdout_combos:     how many (material, context) pairs are reserved as
                        compositional eval (drawn from in-pool values only).
    template_subset:    optional subset of template indices to use.
    """

    AXIS_NAMES: Tuple[str, ...] = ("material", "color", "pattern", "context", "style")

    def __init__(
        self,
        seed: int = 0,
        holdout_per_axis: int = 2,
        holdout_combos: int = 8,
        template_subset: Optional[Sequence[int]] = None,
    ):
        self.seed = seed
        self._rng = random.Random(seed)

        # Deterministic split RNG, independent from sampling RNG.
        split_rng = random.Random(seed * 1009 + 17)
        self._train_values: Dict[str, List[str]] = {}
        self._ood_values: Dict[str, List[str]] = {}
        for axis in self.AXIS_NAMES:
            values = list(AXES[axis])
            split_rng.shuffle(values)
            n_ood = max(1, min(holdout_per_axis, len(values) // 3))
            self._ood_values[axis] = sorted(values[:n_ood])
            self._train_values[axis] = sorted(values[n_ood:])

        # Compositional held-out pairs: (material, context) drawn from
        # in-pool values; these specific pairs are forbidden during training.
        materials = self._train_values["material"]
        contexts = self._train_values["context"]
        all_pairs = [(m, c) for m in materials for c in contexts]
        split_rng.shuffle(all_pairs)
        n_comp = min(holdout_combos, len(all_pairs))
        self._comp_pairs: List[Tuple[str, str]] = all_pairs[:n_comp]
        self._comp_pair_set = set(self._comp_pairs)

        if template_subset is None:
            self._template_indices = list(range(len(TEMPLATES)))
        else:
            self._template_indices = [int(i) for i in template_subset]

    # ------------------------------------------------------------------
    # Public sampling API
    # ------------------------------------------------------------------

    def sample_train(self) -> PromptSample:
        """Sample one training prompt.

        Samples each axis independently from `_train_values` and rejects
        any sample whose (material, context) hits a comp hold-out pair
        (rejection sampling, expected reject rate ≪ 1).
        """
        for _ in range(32):
            axes = {ax: self._rng.choice(self._train_values[ax])
                    for ax in self.AXIS_NAMES}
            if (axes["material"], axes["context"]) not in self._comp_pair_set:
                break
        # Fallback (should be unreachable): perturb context
        if (axes["material"], axes["context"]) in self._comp_pair_set:
            allowed = [c for c in self._train_values["context"]
                       if (axes["material"], c) not in self._comp_pair_set]
            if allowed:
                axes["context"] = self._rng.choice(allowed)

        t_idx = self._rng.choice(self._template_indices)
        text = TEMPLATES[t_idx].format(**axes)
        return PromptSample(text=text, axes=axes, template_idx=t_idx, split="train")

    def sample_batch(self, batch_size: int) -> List[PromptSample]:
        return [self.sample_train() for _ in range(batch_size)]

    # ------------------------------------------------------------------
    # Evaluation iterators
    # ------------------------------------------------------------------

    def iter_eval_id(self, n: int = 200, seed: int = 12345) -> List[PromptSample]:
        """In-distribution eval: same axes as training, fresh RNG."""
        rng = random.Random(seed)
        out: List[PromptSample] = []
        for _ in range(n):
            for _ in range(32):
                axes = {ax: rng.choice(self._train_values[ax])
                        for ax in self.AXIS_NAMES}
                if (axes["material"], axes["context"]) not in self._comp_pair_set:
                    break
            t_idx = rng.choice(self._template_indices)
            out.append(PromptSample(
                text=TEMPLATES[t_idx].format(**axes),
                axes=axes, template_idx=t_idx, split="eval_id",
            ))
        return out

    def iter_eval_comp(self, seeds_per_pair: int = 4,
                       seed: int = 22345) -> List[PromptSample]:
        """Compositional eval: each held-out (material, context) pair, with
        random other axes."""
        rng = random.Random(seed)
        out: List[PromptSample] = []
        for material, context in self._comp_pairs:
            for _ in range(seeds_per_pair):
                axes = {
                    "material": material,
                    "context": context,
                    "color": rng.choice(self._train_values["color"]),
                    "pattern": rng.choice(self._train_values["pattern"]),
                    "style": rng.choice(self._train_values["style"]),
                }
                t_idx = rng.choice(self._template_indices)
                out.append(PromptSample(
                    text=TEMPLATES[t_idx].format(**axes),
                    axes=axes, template_idx=t_idx, split="eval_comp",
                ))
        return out

    def iter_eval_freeform(self) -> List[PromptSample]:
        """Free-form eval: hand-written user-style prompts that violate
        BOTH the axis vocabulary AND the template grammar. Tests whether
        SDA preserves SDXL's open-vocabulary text conditioning at test time.
        Use this for the headline figure showing controllability."""
        return [
            PromptSample(text=p, axes={}, template_idx=-1, split="eval_freeform")
            for p in FREEFORM_EVAL_PROMPTS
        ]

    def iter_eval_ood(self, n_per_axis: int = 8,
                      seed: int = 32345) -> List[PromptSample]:
        """OOD eval: each prompt mutates exactly ONE axis to its held-out
        values, keeping all other axes in-distribution. Tests whether SDA
        generalizes to unseen axis values without losing prompt control."""
        rng = random.Random(seed)
        out: List[PromptSample] = []
        for axis in self.AXIS_NAMES:
            ood_vals = self._ood_values[axis]
            if not ood_vals:
                continue
            for _ in range(n_per_axis):
                axes = {ax: rng.choice(self._train_values[ax])
                        for ax in self.AXIS_NAMES}
                axes[axis] = rng.choice(ood_vals)
                t_idx = rng.choice(self._template_indices)
                out.append(PromptSample(
                    text=TEMPLATES[t_idx].format(**axes),
                    axes=axes, template_idx=t_idx, split="eval_ood",
                ))
        return out

    # ------------------------------------------------------------------
    # Introspection / persistence
    # ------------------------------------------------------------------

    def describe(self) -> Dict:
        """Return a JSON-serializable description of the split — log this
        once at the start of training so checkpoints can be reproduced."""
        # 计算训练侧组合数（multinomial 上界，未扣除 comp hold-out 对）
        n_combos = 1
        for axis in self.AXIS_NAMES:
            n_combos *= max(1, len(self._train_values[axis]))
        active_families = {
            fam: [i for i in idxs if i in self._template_indices]
            for fam, idxs in TEMPLATE_FAMILIES.items()
        }
        return {
            "seed": self.seed,
            "train_values": self._train_values,
            "ood_values": self._ood_values,
            "comp_pairs": [list(p) for p in self._comp_pairs],
            "templates": [TEMPLATES[i] for i in self._template_indices],
            "template_families": active_families,
            "n_train_axis_combos": n_combos,
            "n_train_combos_after_holdout": max(0, n_combos - len(self._comp_pairs)
                * max(1, len(self._train_values["color"]))
                * max(1, len(self._train_values["pattern"]))
                * max(1, len(self._train_values["style"]))),
            "n_total_train_prompts_estimate": (
                n_combos * len(self._template_indices)
            ),
        }

    def save_split(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.describe(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Convenience helper for the trainer
# ---------------------------------------------------------------------------

def build_default_pool(seed: int = 42) -> PromptPool:
    """Default configuration used by `train_lora_sdxl.py`."""
    return PromptPool(
        seed=seed,
        holdout_per_axis=2,
        holdout_combos=8,
    )


def _format_int(n: int) -> str:
    """Pretty-print large integers, e.g. 1234567 -> '1.23 M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n / 1_000:.2f} k"
    return str(n)


def _print_section(title: str, char: str = "=") -> None:
    print(f"\n{char * 3} {title} {char * 3}")


if __name__ == "__main__":
    prompt_pool = PromptPool(
        seed=42,
        holdout_per_axis=2,
        holdout_combos=8,
    )

    pool_sample = prompt_pool.sample_train()
    prompts = [pool_sample.text]
    # Smoke test for prompt_pool.py.
    # Prints, in order:
    #   1. Axis cardinalities (train side and OOD side)
    #   2. Combinatorial size estimate
    #   3. Held-out compositional pairs
    #   4. One render from EACH template family on the SAME axis values
    #      (so the user can compare surface forms directly)
    #   5. 8 train samples drawn from the live sampler
    #   6. One sample from each eval split

    pool = build_default_pool(seed=42)
    desc = pool.describe()

    # 1. Axis cardinalities
    _print_section("1. Axis cardinalities")
    for axis in pool.AXIS_NAMES:
        n_train = len(desc["train_values"][axis])
        n_ood = len(desc["ood_values"][axis])
        total = n_train + n_ood
        print(f"  {axis:10s}: train={n_train:3d}  ood={n_ood:2d}  total={total:3d}")

    # 2. Combinatorial size
    _print_section("2. Combinatorial size estimate")
    n_axis_combos = desc["n_train_axis_combos"]
    n_after_holdout = desc["n_train_combos_after_holdout"]
    n_total = desc["n_total_train_prompts_estimate"]
    n_templates = len(desc["templates"])
    print(f"  axis combinations          : {_format_int(n_axis_combos)}")
    print(f"  - reserved as comp holdout : "
          f"{_format_int(n_axis_combos - n_after_holdout)}")
    print(f"  - usable for training      : {_format_int(n_after_holdout)}")
    print(f"  templates active           : {n_templates}")
    print(f"  total prompt strings       : {_format_int(n_total)} "
          f"(axis × templates)")
    print(f"  → log2 H(c) ≈ "
          f"{(n_total).bit_length() - 1} bits "
          f"(diversity entropy upper bound)")

    # 3. Held-out compositional pairs
    _print_section("3. Held-out (material, context) pairs")
    for m, c in desc["comp_pairs"]:
        print(f"  ({m}, {c})")

    # 4. One render per template family on identical axis values
    _print_section("4. Same axes rendered through each template family")
    fixed_axes = {
        "material": "rusted metal",
        "color": "olive green",
        "pattern": "irregular stripes",
        "context": "in a dense forest",
        "style": "weathered",
    }
    print(f"  axes = {fixed_axes}")
    for fam, idxs in desc["template_families"].items():
        if not idxs:
            continue
        print(f"\n  -- {fam} --")
        for i in idxs:
            text = TEMPLATES[i].format(**fixed_axes)
            print(f"    [#{i:02d}] {text}")

    # 5. Live sampler output
    _print_section("5. 8 live train samples")
    rng = random.Random(0)  # only for reproducible smoke test ordering
    for k in range(8):
        s = pool.sample_train()
        print(f"  [#{k} comp t={s.template_idx:02d}]  {s.text}")

    # 6. One sample from each eval split
    _print_section("6. One sample from each evaluation split")
    print("  -- eval_id (in-distribution) --")
    print(f"    {pool.iter_eval_id(n=1)[0].text}")
    print("  -- eval_comp (compositional held-out) --")
    print(f"    {pool.iter_eval_comp(seeds_per_pair=1)[0].text}")
    print("  -- eval_ood (axis OOD) --")
    print(f"    {pool.iter_eval_ood(n_per_axis=1)[0].text}")
    print("  -- eval_freeform (user-style natural language) --")
    print(f"    {pool.iter_eval_freeform()[0].text}")
