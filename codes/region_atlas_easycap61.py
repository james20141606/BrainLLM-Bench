"""
Region atlas for Brennan 2019 EEG (61-channel easycap M10).

Maps numbered channels (1..61) to 6 anatomical regions based on
3D coordinates from easycapM10-acti61_elec.sfp.

Convention (sfp file): x = lateral (+R/-L), y = anterior-posterior
(+ant/-post), z = vertical. Splits:
  frontal:           y >  1.5
  parieto-occipital: y < -1.5
  temporal:          |y| <= 1.5 AND |x| > 4 (lateral)

Hemisphere: x > 0 right, x < 0 left, x ≈ 0 midline (assigned to L by
convention to keep parity with Neuromag/KIT atlases).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path("<REPO_ROOT>")
SFP_PATH = PROJ / "data/brennan_2019_meta/easycapM10-acti61_elec.sfp"


def get_easycap61_region_masks(channel_names: list[str]) -> dict[str, np.ndarray]:
    """Return dict region_name -> bool mask of len(channel_names)."""
    sfp = pd.read_csv(SFP_PATH, sep="\t", header=None,
                       names=["label", "x", "y", "z"])
    pos = {str(int(r["label"])): np.array([r["x"], r["y"], r["z"]])
           for _, r in sfp.iterrows()}

    masks = {
        "frontal_L": np.zeros(len(channel_names), dtype=bool),
        "frontal_R": np.zeros(len(channel_names), dtype=bool),
        "temporal_L": np.zeros(len(channel_names), dtype=bool),
        "temporal_R": np.zeros(len(channel_names), dtype=bool),
        "parieto_occipital_L": np.zeros(len(channel_names), dtype=bool),
        "parieto_occipital_R": np.zeros(len(channel_names), dtype=bool),
    }

    for ci, ch in enumerate(channel_names):
        if ch not in pos:
            continue
        x, y, z = pos[ch]
        side = "R" if x > 0 else "L"   # midline → L
        if y > 1.5:
            masks[f"frontal_{side}"][ci] = True
        elif y < -1.5:
            masks[f"parieto_occipital_{side}"][ci] = True
        elif abs(x) > 4:
            masks[f"temporal_{side}"][ci] = True
        else:
            # Central / midline (Cz etc.) — assign to frontal_L per convention
            masks["frontal_L"][ci] = True

    return masks


if __name__ == "__main__":
    # Print a summary
    chs = [str(i) for i in range(1, 62)]
    masks = get_easycap61_region_masks(chs)
    print(f"Total channels: {len(chs)}")
    print(f"Region coverage:")
    for region, m in masks.items():
        print(f"  {region:22s}: {m.sum():2d} channels")
    total_assigned = sum(m.sum() for m in masks.values())
    print(f"  TOTAL ASSIGNED: {total_assigned}")
