"""
Paper 4 — KIT/Yokogawa 208-channel sensor → region atlas.

Mirrors the interface of region_atlas.py (Neuromag-306) so that
compute_rdms.py / stage1_streaming_rsa.py can switch atlases via the
--channels_file / --regions plumbing.

Six anatomical sensor groupings derived once from the KIT-AD layout
(centered head-relative 2-D positions):

    frontal_L  / frontal_R           y > +0.12
    temporal_L / temporal_R          |x| > 0.18 and -0.15 ≤ y ≤ +0.12
    parieto_occipital_L / _R         y < -0.15
    central                          everything else (sensorimotor / midline)

Channel naming follows the BIDS export "MEG 001" .. "MEG 208" with a single
internal whitespace; whitespace is tolerated for matching.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np


# ---- Channel lists per region (derived once from MNE 'KIT-AD' layout) ----
_FRONTAL_L_KIT = [
    "MEG 009", "MEG 017", "MEG 025", "MEG 026", "MEG 036", "MEG 037",
    "MEG 039", "MEG 040", "MEG 041", "MEG 043", "MEG 045", "MEG 047",
    "MEG 048", "MEG 049", "MEG 051", "MEG 054", "MEG 070", "MEG 072",
    "MEG 082", "MEG 083", "MEG 084", "MEG 085", "MEG 086", "MEG 087",
    "MEG 088", "MEG 093", "MEG 095", "MEG 097", "MEG 102", "MEG 103",
    "MEG 107", "MEG 109", "MEG 111", "MEG 112", "MEG 134", "MEG 135",
    "MEG 168",
]
_FRONTAL_R_KIT = [
    "MEG 007", "MEG 024", "MEG 027", "MEG 028", "MEG 029", "MEG 030",
    "MEG 032", "MEG 042", "MEG 044", "MEG 046", "MEG 059", "MEG 060",
    "MEG 062", "MEG 063", "MEG 073", "MEG 074", "MEG 075", "MEG 078",
    "MEG 089", "MEG 090", "MEG 091", "MEG 092", "MEG 094", "MEG 096",
    "MEG 105", "MEG 106", "MEG 108", "MEG 110", "MEG 114", "MEG 121",
    "MEG 123", "MEG 124", "MEG 125", "MEG 130", "MEG 144", "MEG 158",
]
_TEMPORAL_L_KIT = [
    "MEG 033", "MEG 038", "MEG 050", "MEG 052", "MEG 055", "MEG 056",
    "MEG 065", "MEG 067", "MEG 071", "MEG 100", "MEG 101", "MEG 104",
    "MEG 113", "MEG 116", "MEG 118", "MEG 120", "MEG 146", "MEG 150",
    "MEG 167",
]
_TEMPORAL_R_KIT = [
    "MEG 057", "MEG 058", "MEG 061", "MEG 076", "MEG 079", "MEG 122",
    "MEG 127", "MEG 128", "MEG 131", "MEG 132", "MEG 138", "MEG 140",
    "MEG 141", "MEG 143", "MEG 155", "MEG 159", "MEG 174", "MEG 189",
    "MEG 191",
]
_PARIETO_OCCIPITAL_L_KIT = [
    "MEG 016", "MEG 053", "MEG 064", "MEG 066", "MEG 098", "MEG 115",
    "MEG 117", "MEG 119", "MEG 136", "MEG 145", "MEG 148", "MEG 149",
    "MEG 151", "MEG 161", "MEG 162", "MEG 163", "MEG 164", "MEG 165",
    "MEG 166", "MEG 177", "MEG 178", "MEG 179", "MEG 180", "MEG 181",
    "MEG 182", "MEG 183", "MEG 184", "MEG 187", "MEG 197", "MEG 199",
    "MEG 200", "MEG 204",
]
_PARIETO_OCCIPITAL_R_KIT = [
    "MEG 080", "MEG 126", "MEG 129", "MEG 137", "MEG 139", "MEG 142",
    "MEG 147", "MEG 153", "MEG 154", "MEG 156", "MEG 157", "MEG 160",
    "MEG 169", "MEG 170", "MEG 171", "MEG 172", "MEG 176", "MEG 185",
    "MEG 186", "MEG 188", "MEG 190", "MEG 193", "MEG 194", "MEG 195",
    "MEG 196", "MEG 201", "MEG 202", "MEG 203", "MEG 205", "MEG 206",
    "MEG 207", "MEG 208",
]
_CENTRAL_KIT = [
    "MEG 001", "MEG 002", "MEG 003", "MEG 004", "MEG 005", "MEG 006",
    "MEG 008", "MEG 010", "MEG 011", "MEG 012", "MEG 013", "MEG 014",
    "MEG 015", "MEG 018", "MEG 019", "MEG 020", "MEG 021", "MEG 022",
    "MEG 023", "MEG 031", "MEG 034", "MEG 035", "MEG 068", "MEG 069",
    "MEG 077", "MEG 081", "MEG 099", "MEG 133", "MEG 152", "MEG 173",
    "MEG 175", "MEG 192", "MEG 198",
]

REGION_TO_CHANNELS_KIT: dict[str, list[str]] = {
    "frontal_L": _FRONTAL_L_KIT,
    "frontal_R": _FRONTAL_R_KIT,
    "temporal_L": _TEMPORAL_L_KIT,
    "temporal_R": _TEMPORAL_R_KIT,
    "parieto_occipital_L": _PARIETO_OCCIPITAL_L_KIT,
    "parieto_occipital_R": _PARIETO_OCCIPITAL_R_KIT,
    "central": _CENTRAL_KIT,
}

REGION_NAMES_KIT = ("frontal_L", "frontal_R",
                    "temporal_L", "temporal_R",
                    "parieto_occipital_L", "parieto_occipital_R",
                    "central")


def _normalize(name: str) -> str:
    """Collapse whitespace; uppercase. 'meg 001' -> 'MEG 001'."""
    return re.sub(r"\s+", " ", name.strip()).upper()


def _channel_to_region_kit() -> dict[str, str]:
    out = {}
    for region, chs in REGION_TO_CHANNELS_KIT.items():
        for nm in chs:
            out[_normalize(nm)] = region
    return out


_CH_TO_REGION_KIT = _channel_to_region_kit()


def get_kit_region_masks(
    channel_names: Iterable[str],
    regions: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    """Build boolean masks selecting KIT channels per region.

    Args:
        channel_names: ordered list of channel names from data array.
            Names may be "MEG 001" or "MEG001"; whitespace tolerated.
        regions: subset of REGION_NAMES_KIT; default returns the six anatomical
            regions (excludes 'central').

    Returns:
        dict region_name -> np.ndarray[bool] of length len(channel_names).
    """
    if regions is None:
        regions = [r for r in REGION_NAMES_KIT if r != "central"]
    regions = list(regions)
    chs = [_normalize(n) for n in channel_names]
    n_ch = len(chs)
    masks = {r: np.zeros(n_ch, dtype=bool) for r in regions}
    for i, name in enumerate(chs):
        # Allow "MEG001" form as well as "MEG 001"
        if " " not in name and name.startswith("MEG"):
            name = "MEG " + name[3:]
        region = _CH_TO_REGION_KIT.get(name)
        if region in masks:
            masks[region][i] = True
    return masks


def get_canonical_kit208_order() -> list[str]:
    """Return MEG 001 .. MEG 208 in numerical order — the BIDS canonical order."""
    return [f"MEG {i:03d}" for i in range(1, 209)]
