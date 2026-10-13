"""
Paper 4 — Static Neuromag-306 sensor → region atlas.

Six anatomically-motivated sensor groupings derived once from the MNE
Vectorview-mag layout (centered scalp coordinates):

    frontal_L           x < 0,   y > +0.12        (left frontal, IFG-relevant)
    frontal_R           x > 0,   y > +0.12        (right frontal)
    temporal_L          |x| > 0.18,  -0.15 ≤ y ≤ +0.12, x < 0   (left lateral, STG-relevant)
    temporal_R          same, x > 0
    parieto_occipital_L x < 0,   y < -0.15
    parieto_occipital_R x > 0,   y < -0.15
    central             everything else (sensorimotor / midline; default unassigned)

Each magnetometer at location MEG<abc>1 also pulls in its two co-located
planar gradiometers MEG<abc>2 and MEG<abc>3 (Neuromag triplet convention).

Use:
    masks = get_neuromag_region_masks(channel_names)
    stg_data = epochs[:, masks["temporal_L"], :]
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np


# ---- Magnetometers per region (derived once from mne Vectorview-mag layout) ----
_FRONTAL_L_MAGS = [
    "MEG0111", "MEG0121", "MEG0131", "MEG0211", "MEG0221",
    "MEG0311", "MEG0321", "MEG0331", "MEG0341",
    "MEG0511", "MEG0521", "MEG0531", "MEG0541",
    "MEG0611", "MEG0641",
]
_FRONTAL_R_MAGS = [
    "MEG0621", "MEG0811", "MEG0821",
    "MEG0911", "MEG0921", "MEG0931", "MEG0941",
    "MEG1011", "MEG1021", "MEG1031",
    "MEG1211", "MEG1221", "MEG1231", "MEG1241",
    "MEG1311", "MEG1321",
    "MEG1411", "MEG1421", "MEG1441",
]
_TEMPORAL_L_MAGS = [
    "MEG0141", "MEG0231", "MEG0241", "MEG0441",
    "MEG1511", "MEG1521", "MEG1541",
    "MEG1611", "MEG1621", "MEG1631",
]
_TEMPORAL_R_MAGS = [
    "MEG1131", "MEG1331", "MEG1341", "MEG1431",
    "MEG2411", "MEG2421", "MEG2441",
    "MEG2611", "MEG2621", "MEG2641",
]
_PARIETO_OCCIPITAL_L_MAGS = [
    "MEG1531", "MEG1641",
    "MEG1711", "MEG1721", "MEG1731", "MEG1741",
    "MEG1911", "MEG1921", "MEG1931", "MEG1941",
    "MEG2011", "MEG2041", "MEG2111", "MEG2141",
]
_PARIETO_OCCIPITAL_R_MAGS = [
    "MEG2021", "MEG2031", "MEG2121", "MEG2131",
    "MEG2311", "MEG2321", "MEG2331", "MEG2341",
    "MEG2431",
    "MEG2511", "MEG2521", "MEG2531", "MEG2541",
    "MEG2631",
]
_CENTRAL_MAGS = [
    "MEG0411", "MEG0421", "MEG0431", "MEG0631",
    "MEG0711", "MEG0721", "MEG0731", "MEG0741",
    "MEG1041", "MEG1111", "MEG1121", "MEG1141",
    "MEG1811", "MEG1821", "MEG1831", "MEG1841",
    "MEG2211", "MEG2221", "MEG2231", "MEG2241",
]

REGION_TO_MAGS: dict[str, list[str]] = {
    "frontal_L": _FRONTAL_L_MAGS,
    "frontal_R": _FRONTAL_R_MAGS,
    "temporal_L": _TEMPORAL_L_MAGS,
    "temporal_R": _TEMPORAL_R_MAGS,
    "parieto_occipital_L": _PARIETO_OCCIPITAL_L_MAGS,
    "parieto_occipital_R": _PARIETO_OCCIPITAL_R_MAGS,
    "central": _CENTRAL_MAGS,
}

REGION_NAMES = ("frontal_L", "frontal_R",
                "temporal_L", "temporal_R",
                "parieto_occipital_L", "parieto_occipital_R",
                "central")

# 102 mags + 204 grads, by Neuromag triplet convention: MEG<abc>1 (mag),
# MEG<abc>2 + MEG<abc>3 (planar gradiometers at same location).
_MEG_PATTERN = re.compile(r"MEG\s*0*(\d{3})(\d)$")


def _normalize_name(name: str) -> str:
    """Strip whitespace; keep canonical 'MEG' + 4 digits form."""
    return name.replace(" ", "").upper().strip()


def _location_code(name: str) -> str | None:
    """Return the 3-digit location code for a Neuromag MEG channel; None for non-MEG."""
    m = _MEG_PATTERN.match(_normalize_name(name))
    return m.group(1) if m else None


def _build_location_to_region() -> dict[str, str]:
    """Invert REGION_TO_MAGS into a location-code → region map."""
    out: dict[str, str] = {}
    for region, mag_names in REGION_TO_MAGS.items():
        for nm in mag_names:
            code = _location_code(nm)
            if code is not None:
                out[code] = region
    return out


_LOCATION_TO_REGION: dict[str, str] = _build_location_to_region()


def get_neuromag_region_masks(
    channel_names: Iterable[str],
    regions: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Build boolean masks selecting channels belonging to each region.

    Args:
        channel_names: ordered list of channel names from the data array.
            Names may be 'MEG0111' or 'MEG 0111' (whitespace tolerated).
        regions: subset of REGION_NAMES; default returns all six anatomical
            regions (excludes 'central').

    Returns:
        dict region_name -> np.ndarray[bool] of length len(channel_names).
        Each True entry indicates that channel (a magnetometer or one of its
        two co-located planar gradiometers) belongs to that region.

    Notes:
        Non-MEG channels (EOG, ECG, EMG, IAS*, …) and any MEG channel whose
        location is not catalogued (i.e. landed in 'central' and the caller
        omits it from `regions`) are False in every returned mask.
    """
    if regions is None:
        regions = [r for r in REGION_NAMES if r != "central"]
    regions = list(regions)

    chs = [_normalize_name(n) for n in channel_names]
    n_ch = len(chs)
    masks = {r: np.zeros(n_ch, dtype=bool) for r in regions}

    for i, name in enumerate(chs):
        code = _location_code(name)
        if code is None:
            continue
        region = _LOCATION_TO_REGION.get(code)
        if region in masks:
            masks[region][i] = True

    return masks


def get_region_for_channel(name: str) -> str | None:
    """Return the region label for a single channel name, or None if unmapped."""
    code = _location_code(name)
    return _LOCATION_TO_REGION.get(code) if code is not None else None


def get_canonical_neuromag306_order() -> list[str]:
    """
    Return the 306 Elekta/Neuromag MEG channel names in canonical triplet order:
        MEG0111 (mag), MEG0112 (grad), MEG0113 (grad),
        MEG0121 (mag), MEG0122 (grad), MEG0123 (grad), …

    This order matches the LibriBrain serialized H5 `channel_names` attribute
    and the BIDS `channels.tsv` row order for any Neuromag-306 BIDS dataset.
    """
    # All magnetometer location codes from the seven regions.
    all_mags = sum(REGION_TO_MAGS.values(), [])
    locs = sorted({_location_code(m) for m in all_mags})  # 102 unique 3-digit codes
    out: list[str] = []
    for loc in locs:
        out.append(f"MEG{loc}1")  # magnetometer
        out.append(f"MEG{loc}2")  # planar gradiometer 1
        out.append(f"MEG{loc}3")  # planar gradiometer 2
    return out
