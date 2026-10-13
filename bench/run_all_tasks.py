"""bench/run_all_tasks.py — populate bench submissions for all 5 tasks
from already-computed paper results, producing gold-standard reference
outputs for the benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "bench/submissions"
RES = PROJ / "results"


def task2_boundary_conditional():
    """Read App.G4 results; produce per-cohort × LLM boundary TSV."""
    src = RES / "exp_boundary_conditional/boundary_conditional_v2.tsv"
    if not src.exists(): return
    df = pd.read_csv(src, sep="\t")
    # MEG-SCANS results in separate file
    src_ms = RES / "exp_boundary_conditional/boundary_megscans.tsv"
    if src_ms.exists():
        df_ms = pd.read_csv(src_ms, sep="\t")
        df = pd.concat([df, df_ms], ignore_index=True)
    pivot = df.pivot_table(index=["cohort", "llm"], columns="condition",
                            values="z").reset_index()
    pivot["delta"] = pivot["medial"] - pivot["onset"]
    pivot["cell_id"] = pivot.apply(
        lambda r: f"{r['cohort'].lower().replace('-', '_')}_{r['llm']}",
        axis=1)
    pivot = pivot.rename(columns={"medial": "z_medial", "onset": "z_onset"})
    pivot[["cell_id", "z_onset", "z_medial", "delta"]].to_csv(
        OUT / "boundary_conditional_gold.tsv", sep="\t", index=False)
    n_pos = (pivot["delta"] > 0).sum()
    print(f"[task2] boundary-conditional: {len(pivot)} cells, "
          f"{n_pos}/{len(pivot)} Zou-Ding-consistent",
          flush=True)


def task3_surprisal_tercile():
    """Per-cohort surprisal tercile weighted-mean active layer."""
    # Already computed in results/exp_source_surprisal_multisubj for MS;
    # other cohorts in stage3/stage4/stage5_surprisal dirs.
    rows = []
    multisubj = RES / "exp_source_surprisal_multisubj/source_surprisal_tercile_multisubj.tsv"
    if multisubj.exists():
        df = pd.read_csv(multisubj, sep="\t")
        for model in df["model"].unique():
            sub = df[df["model"] == model]
            # weighted-mean active layer per tercile
            for t in ["low", "mid", "high"]:
                ss = sub[sub["tercile"] == t]
                w = ss["z_rho"].clip(lower=0)
                if w.sum() == 0: continue
                wm = float((ss["layer"] * w).sum() / w.sum())
                rows.append({"cell_id": f"meg_scans_{model}",
                             "tercile": t,
                             "weighted_mean_layer": wm})
    pd.DataFrame(rows).to_csv(OUT / "surprisal_tercile_gold.tsv",
                                sep="\t", index=False)
    print(f"[task3] surprisal-tercile: {len(rows)} rows", flush=True)


def task4_sae_feature_transfer():
    """CITE-SAE top-50 overlap with LibriBrain anchor."""
    sae_dir = RES / "method_cite_sae"
    if not sae_dir.exists(): return
    cohorts = {}
    for f in sae_dir.glob("sae_features_*.tsv"):
        name = f.stem.replace("sae_features_", "")
        df = pd.read_csv(f, sep="\t")
        if "z_rho_mean" in df.columns:
            df = df.rename(columns={"z_rho_mean": "z_rho"})
        cohorts[name] = df
    if "libribrain" not in cohorts: return
    anchor_top = set(cohorts["libribrain"]
                       .nlargest(50, "z_rho")["sae_feature_idx"])
    rows = []
    for name, df in cohorts.items():
        top = set(df.nlargest(50, "z_rho")["sae_feature_idx"])
        overlap = len(top & anchor_top)
        rows.append({"cell_id": f"sae_{name}",
                     "top50_overlap_with_LB": overlap})
    pd.DataFrame(rows).to_csv(OUT / "sae_feature_transfer_gold.tsv",
                                sep="\t", index=False)
    print(f"[task4] SAE-feature-transfer: {len(rows)} cohorts", flush=True)


def task5_pfll_layer_saturation():
    """L_sat depth % per (cohort, LLM) from existing PFLL data."""
    # Use the bench best-layer submissions as proxy: L_sat ≈ best_layer
    # (since peak alignment ≈ L_sat for 95% threshold).
    rows = []
    for f in OUT.glob("*_best_layer.tsv"):
        if "persubj" in f.name: continue
        df = pd.read_csv(f, sep="\t")
        if df.empty or "cell_id" not in df.columns: continue
        cell = df.iloc[0]
        # Approximate L_sat as best_layer; depth % = layer/(n_layers-1)
        L = int(cell["predicted_best_layer"])
        # Heuristic n_layers from LLM name
        if "gpt2-medium" in cell["cell_id"]: n_layers = 25
        elif "german-gpt2" in cell["cell_id"] or "dbmdz" in cell["cell_id"]:
            n_layers = 13
        else: n_layers = 13
        depth_pct = 100 * L / max(1, n_layers - 1)
        rows.append({"cell_id": cell["cell_id"],
                     "L_sat_proxy": L,
                     "L_sat_depth_pct": depth_pct})
    pd.DataFrame(rows).to_csv(OUT / "pfll_saturation_gold.tsv",
                                sep="\t", index=False)
    print(f"[task5] PFLL-saturation: {len(rows)} cells", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    task2_boundary_conditional()
    task3_surprisal_tercile()
    task4_sae_feature_transfer()
    task5_pfll_layer_saturation()
    print("\n[done] all 4 derived tasks populated in bench/submissions/",
          flush=True)


if __name__ == "__main__":
    main()
