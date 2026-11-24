"""The detection vocabulary: asset classes, their SAM 3 text prompts and their mount type.

A vocabulary file maps class keys to {"prompts": [...], "mount": "ground" | "pole" | "surface"}; keys that
start with "_" are comments. Mount types decide which point of a detection touches the ground:
    ground   the bottom of the mask is the ground contact
    pole     the object hangs on a pole; the contact point is borrowed from the pole below it
    surface  the object lies flush with the ground; the mask centroid is used
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CONFIGS = Path(__file__).resolve().parent / "configs"
DEFAULT_VOCABULARY = CONFIGS / "prompts.json"
DEFAULT_PARAMS = CONFIGS / "fusion_params.json"


@dataclass
class Vocabulary:
    prompts: list[str]                 # all prompts, in file order
    class_of: dict[str, str]           # prompt -> class key
    rank_of: dict[str, int]            # prompt -> position within its class (0 = primary prompt)
    mount_of: dict[str, str]           # class key -> mount type

    @classmethod
    def load(cls, path=DEFAULT_VOCABULARY) -> Vocabulary:
        voc = json.loads(Path(path).read_text())
        prompts, class_of, rank_of, mount_of = [], {}, {}, {}
        for key, v in voc.items():
            if key.startswith("_"):
                continue
            mount_of[key] = v.get("mount", "ground")
            for r, p in enumerate(v["prompts"]):
                if p in class_of:
                    raise ValueError(f"prompt {p!r} is used by two classes ({class_of[p]}, {key})")
                prompts.append(p)
                class_of[p] = key
                rank_of[p] = r
        return cls(prompts, class_of, rank_of, mount_of)
