"""
Paper 4 - TemporalAlign
Step 1: Compute MEG Representational Dissimilarity Matrices (RDMs) per time window.

For each sliding time window, computes a [N_trials x N_trials] RDM using
1 - Pearson correlation between trial MEG patterns (channels as features).

Also supports frequency-band-specific RDMs via bandpass filtering.

Usage:
    python compute_rdms.py \
        --meg_dir /data/preprocessed/meg_epochs \
        --output_dir /data/preprocessed/rdms \
        --dataset libribrain \
        --window_ms 50 \
        --step_ms 20 \
        --tmin -0.2 \
        --tmax 0.8 \
        --freq_bands all

Requirements: mne>=1.0, numpy, scipy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.spatial.distance as distance

sys.path.insert(0, str(Path(__file__).resolve().parent))
from region_atlas import (  # noqa: E402
    REGION_NAMES,
    get_canonical_neuromag306_order,
    get_neuromag_region_masks,
)


# --------------------------------------------------------------------------- #
# Frequency band definitions (Hz)
# --------------------------------------------------------------------------- #
FREQ_BANDS = {
    "broadband": (1, 150),
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 80),
}


def load_epochs_mne(meg_dir: Path, dataset: str):
    """
    Load pre-epoched MEG data from disk.

    Returns:
        epochs_data: np.ndarray [N_trials, N_channels, N_times]
        sfreq: float  (sampling frequency in Hz)
        times: np.ndarray [N_times]  (time vector in seconds)
        trial_labels: list[str]  (word/stimulus label per trial)
    """
    try:
        import mne

        epoch_file = meg_dir / f"{dataset}_epochs-epo.fif"
        if not epoch_file.exists():
            # Fall back to .pt format (preprocessed by existing pipeline)
            return _load_epochs_pt(meg_dir, dataset)

        epochs = mne.read_epochs(str(epoch_file), preload=True, verbose=False)
        epochs_data = epochs.get_data()  # [N, C, T]
        sfreq = epochs.info["sfreq"]
        times = epochs.times
        trial_labels = list(epochs.metadata["word"]) if epochs.metadata is not None else [str(i) for i in range(len(epochs))]
        return epochs_data, sfreq, times, trial_labels

    except ImportError:
        print("MNE not available; falling back to .pt loader")
        return _load_epochs_pt(meg_dir, dataset)


def _load_epochs_pt(meg_dir: Path, dataset: str):
    """Load epochs from .pt or .npy files produced by preprocessing scripts."""
    # Try .npy first (more robust for large files)
    npy_file = meg_dir / "epochs_data.npy"
    meta_file = meg_dir / "metadata.json"
    words_file = meg_dir / "stimuli_words.txt"

    if npy_file.exists() and meta_file.exists():
        import json
        epochs_data = np.load(npy_file, mmap_mode="r")
        with open(meta_file) as f:
            meta = json.load(f)
        sfreq = float(meta["sfreq"])
        times = np.linspace(meta["epoch_tmin"], meta["epoch_tmax"], epochs_data.shape[2])
        if words_file.exists():
            with open(words_file) as f:
                trial_labels = [line.strip() for line in f if line.strip()]
        else:
            trial_labels = [str(i) for i in range(len(epochs_data))]
        return epochs_data, sfreq, times, trial_labels

    # Fallback to .pt
    import torch
    pt_file = meg_dir / f"{dataset}_epochs.pt"
    if not pt_file.exists():
        raise FileNotFoundError(f"No epoch file found at {meg_dir}. Run preprocess_meg.py first.")
    data = torch.load(pt_file, map_location="cpu", weights_only=False)
    epochs_data = data["epochs"].numpy()  # [N, C, T]
    sfreq = float(data["sfreq"])
    times = data["times"].numpy()
    trial_labels = data.get("labels", [str(i) for i in range(len(epochs_data))])
    return epochs_data, sfreq, times, trial_labels


def bandpass_filter(epochs_data: np.ndarray, sfreq: float, fmin: float, fmax: float) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter to epochs.

    Args:
        epochs_data: [N, C, T]
        sfreq: sampling frequency
        fmin, fmax: passband limits in Hz

    Returns:
        filtered: [N, C, T]
    """
    try:
        import mne

        # Use MNE's filter which handles edge effects properly
        data_64 = epochs_data.astype(np.float64).reshape(-1, epochs_data.shape[-1])
        filtered = mne.filter.filter_data(
            data_64,
            sfreq=sfreq,
            l_freq=fmin,
            h_freq=fmax,
            method="iir",
            iir_params={"order": 4, "ftype": "butter"},
            verbose=False,
        )
        return filtered.reshape(epochs_data.shape).astype(np.float32)
    except ImportError:
        # Fallback: scipy butterworth
        from scipy import signal

        b, a = signal.butter(4, [fmin / (sfreq / 2), fmax / (sfreq / 2)], btype="band")
        return signal.filtfilt(b, a, epochs_data, axis=-1)


def compute_rdm_at_window(
    epochs_data: np.ndarray,
    t_start_idx: int,
    t_end_idx: int,
    region_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute RDM for a time window [t_start_idx : t_end_idx].

    MEG pattern per trial = flattened [channels x time] vector.
    Distance metric = 1 - Pearson correlation.

    Args:
        epochs_data: [N_trials, N_channels, N_times]
        t_start_idx, t_end_idx: time sample indices
        region_mask: optional boolean array of length N_channels selecting
            a sensor subset. None = use all channels.

    Returns:
        rdm: [N_trials, N_trials] symmetric matrix (upper triangle = valid)
    """
    window = epochs_data[:, :, t_start_idx:t_end_idx]  # [N, C, W]
    if region_mask is not None:
        window = window[:, region_mask, :]
    patterns = window.reshape(len(window), -1)  # [N, C_sub*W]
    rdm = distance.squareform(distance.pdist(patterns, metric="correlation"))
    return rdm.astype(np.float32)


def compute_all_rdms(
    epochs_data: np.ndarray,
    times: np.ndarray,
    sfreq: float,
    tmin: float = -0.2,
    tmax: float = 0.8,
    window_ms: float = 50.0,
    step_ms: float = 20.0,
    region_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window across [tmin, tmax] and compute one RDM per step.

    Returns:
        rdms: [T_windows, N_trials, N_trials]
        window_centers: [T_windows] in seconds
    """
    window_samples = int(window_ms / 1000.0 * sfreq)
    step_samples = int(step_ms / 1000.0 * sfreq)

    t_start = np.searchsorted(times, tmin)
    t_end = np.searchsorted(times, tmax)

    rdms = []
    centers = []
    idx = t_start
    while idx + window_samples <= t_end:
        rdm = compute_rdm_at_window(epochs_data, idx, idx + window_samples,
                                     region_mask=region_mask)
        rdms.append(rdm)
        center_time = times[idx + window_samples // 2]
        centers.append(center_time)
        idx += step_samples

    return np.stack(rdms), np.array(centers)


def _session_id(row) -> str:
    """Build a stable session identifier from trial-metadata columns."""
    parts = [
        f"sub-{row['subject']}",
        f"task-{row['task']}",
        f"ses-{row['session']}",
        f"run-{row['run']}",
    ]
    return "_".join(parts)


def _process_one_session(
    sess_name: str,
    sess_idx_in_pool: np.ndarray,
    epochs_path: Path,
    epochs_shape: tuple[int, int, int],
    epochs_dtype,
    sfreq: float,
    times: np.ndarray,
    bands_to_run: list[str],
    region_masks: dict,
    output_root: Path,
    dataset: str,
    window_ms: float,
    step_ms: float,
    tmin: float,
    tmax: float,
    n_trials_original: int,
) -> dict:
    """Compute RDMs for one session, all requested (region, band) cells.

    region_masks: dict mapping region_name -> np.ndarray[bool] | None.
        A None mask means 'all channels'. Iterate this dict to fan out
        across regions while paying the per-session epoch read just once.
    """
    epochs_mm = np.memmap(epochs_path, dtype=epochs_dtype, mode="r",
                          shape=epochs_shape)
    sess_epochs = np.asarray(epochs_mm[sess_idx_in_pool], dtype=np.float32)
    n_trials = sess_epochs.shape[0]

    summary = {"session_id": sess_name, "n_trials": int(n_trials), "cells": {}}

    # Bandpass once per band, reuse across regions.
    for band_name in bands_to_run:
        fmin, fmax = FREQ_BANDS[band_name]
        if band_name == "broadband":
            band_data = sess_epochs
        else:
            band_data = bandpass_filter(sess_epochs, sfreq, fmin, fmax)

        for region_name, region_mask in region_masks.items():
            rdms, window_centers = compute_all_rdms(
                band_data, times, sfreq,
                tmin=tmin, tmax=tmax,
                window_ms=window_ms, step_ms=step_ms,
                region_mask=region_mask,
            )

            cell_dir = (output_root / dataset / sess_name / band_name
                        if region_name == "all_channels"
                        else output_root / dataset / sess_name / region_name / band_name)
            cell_dir.mkdir(parents=True, exist_ok=True)
            np.save(cell_dir / "rdms.npy", rdms)
            np.save(cell_dir / "window_centers.npy", window_centers)
            np.save(cell_dir / "trial_idx_in_pool.npy", sess_idx_in_pool)

            meta = {
                "dataset": dataset,
                "session_id": sess_name,
                "band": band_name,
                "region": region_name,
                "n_channels_used": int(region_mask.sum()) if region_mask is not None
                                    else int(epochs_shape[1]),
                "fmin": fmin, "fmax": fmax,
                "n_trials": int(n_trials),
                "n_trials_original": int(n_trials_original),
                "n_windows": int(len(rdms)),
                "window_ms": window_ms, "step_ms": step_ms,
                "tmin": tmin, "tmax": tmax,
                "sfreq": sfreq,
            }
            with open(cell_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)
            summary["cells"][f"{region_name}/{band_name}"] = {
                "rdms_shape": list(rdms.shape),
                "out_dir": str(cell_dir),
            }

        # Drop band copy if it was a separate buffer
        if band_data is not sess_epochs:
            del band_data

    del sess_epochs
    return summary


def run_per_session_mode(
    args,
    epochs_data: np.ndarray,
    sfreq: float,
    times: np.ndarray,
    bands_to_run: list[str],
    region_masks: dict,
    output_dir: Path,
) -> None:
    """Group trials by session and compute one RDM per session × band × region."""
    import pandas as pd

    meta_path = (Path(args.trial_metadata_file) if args.trial_metadata_file
                 else Path(args.meg_dir) / "trial_metadata.tsv")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"--per_session requires {meta_path} (or pass --trial_metadata_file). "
        )
    print(f"Per-session mode: reading {meta_path}")
    meta = pd.read_csv(meta_path, sep="\t")
    if "trial_index" not in meta.columns:
        meta = meta.reset_index().rename(columns={"index": "trial_index"})

    n_epochs = epochs_data.shape[0]
    bad = (meta["trial_index"] < 0) | (meta["trial_index"] >= n_epochs)
    if bad.any():
        raise ValueError(
            f"trial_metadata has {int(bad.sum())} trial_index values outside "
            f"[0, {n_epochs}). Subsetting must preserve absolute indices."
        )
    if len(meta) < n_epochs:
        print(f"  Subset metadata: {len(meta)}/{n_epochs} trials selected")

    # Build session_id and group trial_index lists.
    meta["__session_id"] = meta.apply(_session_id, axis=1)
    sess_groups = meta.groupby("__session_id", sort=True)
    sess_items = [(name, np.array(group["trial_index"].values, dtype=np.int64))
                  for name, group in sess_groups]
    print(f"  Found {len(sess_items)} sessions; "
          f"trials/session: min={min(len(idx) for _, idx in sess_items)}, "
          f"max={max(len(idx) for _, idx in sess_items)}, "
          f"total={sum(len(idx) for _, idx in sess_items)}")

    n_trials_original = epochs_data.shape[0]
    epochs_path = Path(args.meg_dir) / "epochs_data.npy"
    epochs_shape = tuple(epochs_data.shape)
    epochs_dtype = epochs_data.dtype

    # Free the parent process's mmap handle so workers don't double-fault on it.
    del epochs_data

    work_args = dict(
        epochs_path=epochs_path,
        epochs_shape=epochs_shape,
        epochs_dtype=epochs_dtype,
        sfreq=sfreq,
        times=times,
        bands_to_run=list(bands_to_run),
        region_masks=region_masks,
        output_root=output_dir,
        dataset=args.dataset,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        tmin=args.tmin,
        tmax=args.tmax,
        n_trials_original=n_trials_original,
    )

    if args.n_jobs and args.n_jobs > 1:
        try:
            from joblib import Parallel, delayed
        except ImportError as e:
            raise SystemExit(
                "--n_jobs > 1 requires joblib. `pip install joblib` or use n_jobs=1."
            ) from e
        print(f"  Parallel: {args.n_jobs} workers")
        results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
            delayed(_process_one_session)(name, idx, **work_args)
            for name, idx in sess_items
        )
    else:
        results = []
        for k, (name, idx) in enumerate(sess_items, start=1):
            print(f"  [{k}/{len(sess_items)}] {name}: n={len(idx)} trials")
            results.append(_process_one_session(name, idx, **work_args))

    summary_path = output_dir / args.dataset / "per_session_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "n_sessions": len(results),
                "regions": list(region_masks.keys()),
                "bands": list(bands_to_run),
                "sessions": results,
            },
            f, indent=2,
        )
    print(f"\nPer-session RDMs written under {output_dir / args.dataset}")
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute MEG RDMs per time window")
    parser.add_argument("--meg_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="libribrain",
                        choices=["libribrain", "meg_masc", "armeni", "brennan"])
    parser.add_argument("--window_ms", type=float, default=50.0,
                        help="Time window width in ms")
    parser.add_argument("--step_ms", type=float, default=20.0,
                        help="Sliding step size in ms")
    parser.add_argument("--tmin", type=float, default=-0.2,
                        help="Epoch start in seconds")
    parser.add_argument("--tmax", type=float, default=0.8,
                        help="Epoch end in seconds")
    parser.add_argument("--freq_bands", nargs="+",
                        default=["broadband"],
                        choices=list(FREQ_BANDS.keys()) + ["all"],
                        help="Which frequency bands to compute RDMs for")
    parser.add_argument("--max_trials", type=int, default=800,
                        help="Max trials for RDM computation (subsample if needed). "
                             "0 = use all trials (WARNING: memory-intensive).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for trial subsampling")
    parser.add_argument("--n_bootstrap", type=int, default=1,
                        help="Number of bootstrap subsamples (1 = single run, >1 = bootstrap mode)")
    parser.add_argument("--bootstrap_size", type=int, default=1000,
                        help="Number of trials per bootstrap subsample")
    parser.add_argument("--region", type=str, default="all_channels",
                        choices=["all_channels", *REGION_NAMES],
                        help="Single sensor region (legacy single-output mode). "
                             "Use --regions for fan-out across multiple regions in "
                             "one invocation.")
    parser.add_argument("--regions", nargs="+", default=None,
                        choices=["all_channels", *REGION_NAMES],
                        help="Fan out across multiple regions in one invocation "
                             "(per-session mode only). Each region's RDMs are written "
                             "to <out>/<dataset>/<session>/<region>/<band>/. "
                             "Overrides --region when set.")
    parser.add_argument("--channels_file", type=str, default=None,
                        help="Optional path to a BIDS channels.tsv. If omitted, "
                             "the canonical Neuromag-306 triplet order is assumed "
                             "(matches LibriBrain serialized H5 layout).")
    parser.add_argument("--per_session", action="store_true",
                        help="Group trials by (subject, session, task, run) from "
                             "trial_metadata.tsv and compute one RDM per session. "
                             "Output layout: <out>/<dataset>/<session_id>/<region>/<band>/. "
                             "Implies --max_trials 0 (no cross-session subsampling).")
    parser.add_argument("--trial_metadata_file", type=str, default=None,
                        help="Path to trial_metadata.tsv. Required when --per_session "
                             "is set. Defaults to <meg_dir>/trial_metadata.tsv.")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel session workers (joblib). "
                             "Only used in --per_session mode.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meg_dir = Path(args.meg_dir)
    print(f"Loading MEG data from {meg_dir} ...")
    epochs_data, sfreq, times, trial_labels = load_epochs_mne(meg_dir, args.dataset)
    print(f"  Loaded: {epochs_data.shape[0]} trials, {epochs_data.shape[1]} channels, "
          f"{epochs_data.shape[2]} time points @ {sfreq} Hz")

    n_trials = epochs_data.shape[0]
    bands_to_run = list(FREQ_BANDS.keys()) if "all" in args.freq_bands else args.freq_bands

    # Resolve channel names (for region masking) once.
    if args.channels_file is not None:
        import pandas as pd
        ch_df = pd.read_csv(args.channels_file, sep="\t")
        ch_names = ch_df.loc[
            ch_df["type"].isin(["MEGMAG", "MEGGRADPLANAR"]), "name"
        ].tolist()
    else:
        ch_names = get_canonical_neuromag306_order()
    if len(ch_names) != epochs_data.shape[1]:
        raise ValueError(
            f"Channel-name list length ({len(ch_names)}) does not match "
            f"epochs_data channel dim ({epochs_data.shape[1]}). "
            f"Pass --channels_file if your data uses a non-canonical layout."
        )

    # Build {region_name: mask_or_None} for the per-session fan-out
    requested_regions = args.regions if args.regions else [args.region]
    if args.regions and not args.per_session:
        raise SystemExit("--regions (multi) requires --per_session. Use --region for single-region legacy paths.")
    region_masks: dict = {}
    for r in requested_regions:
        if r == "all_channels":
            region_masks[r] = None
        else:
            mask = get_neuromag_region_masks(ch_names, regions=[r])[r]
            region_masks[r] = mask
            print(f"  Region '{r}': selecting {mask.sum()}/{len(ch_names)} channels")

    if args.per_session:
        # ---- Per-session mode (Stage 1) ---- #
        run_per_session_mode(
            args=args,
            epochs_data=epochs_data,
            sfreq=sfreq,
            times=times,
            bands_to_run=bands_to_run,
            region_masks=region_masks,
            output_dir=output_dir,
        )
        return

    # Single-region back-compat: the legacy bootstrap/single-run paths only
    # support a single region. Pick the first one (which is `--region` if
    # `--regions` wasn't passed).
    region_mask = region_masks[requested_regions[0]]

    if args.n_bootstrap > 1:
        # ---- Bootstrap subsampling mode ---- #
        print(f"\n=== Bootstrap mode: K={args.n_bootstrap}, "
              f"size={args.bootstrap_size} trials each ===")
        rng = np.random.RandomState(args.seed)

        for k in range(args.n_bootstrap):
            print(f"\n--- Bootstrap {k+1}/{args.n_bootstrap} ---")
            selected_idx = np.sort(rng.choice(
                n_trials, min(args.bootstrap_size, n_trials), replace=False
            ))
            sub_epochs = epochs_data[selected_idx]
            sub_labels = [trial_labels[i] for i in selected_idx]

            # Save indices for this bootstrap
            boot_dir = output_dir / args.dataset / f"bootstrap_{k}"
            boot_dir.mkdir(parents=True, exist_ok=True)
            np.save(boot_dir / "selected_trial_idx.npy", selected_idx)
            np.save(boot_dir / "trial_labels.npy", np.array(sub_labels))

            for band_name in bands_to_run:
                fmin, fmax = FREQ_BANDS[band_name]
                print(f"  Band: {band_name} ({fmin}-{fmax} Hz)")

                if band_name == "broadband":
                    data = sub_epochs
                else:
                    data = bandpass_filter(sub_epochs, sfreq, fmin, fmax)

                rdms, window_centers = compute_all_rdms(
                    data, times, sfreq,
                    tmin=args.tmin, tmax=args.tmax,
                    window_ms=args.window_ms, step_ms=args.step_ms,
                    region_mask=region_mask,
                )

                band_dir = (boot_dir / band_name
                            if args.region == "all_channels"
                            else boot_dir / args.region / band_name)
                band_dir.mkdir(parents=True, exist_ok=True)
                np.save(band_dir / "rdms.npy", rdms)
                np.save(band_dir / "window_centers.npy", window_centers)

                metadata = {
                    "dataset": args.dataset,
                    "band": band_name,
                    "region": args.region,
                    "n_channels_used": int(region_mask.sum()) if region_mask is not None else int(epochs_data.shape[1]),
                    "fmin": fmin, "fmax": fmax,
                    "n_trials": len(selected_idx),
                    "n_trials_original": n_trials,
                    "bootstrap_k": k,
                    "bootstrap_size": args.bootstrap_size,
                    "seed": args.seed,
                    "n_windows": len(rdms),
                    "window_ms": args.window_ms, "step_ms": args.step_ms,
                    "tmin": args.tmin, "tmax": args.tmax,
                    "sfreq": sfreq,
                }
                with open(band_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

        print(f"\nBootstrap RDMs saved to {output_dir / args.dataset}")

    else:
        # ---- Single run mode (original behavior) ---- #
        selected_idx = None
        if args.max_trials > 0 and n_trials > args.max_trials:
            rng = np.random.RandomState(args.seed)
            selected_idx = np.sort(rng.choice(n_trials, args.max_trials, replace=False))
            epochs_data = epochs_data[selected_idx]
            trial_labels = [trial_labels[i] for i in selected_idx]
            print(f"  Subsampled to {args.max_trials} trials (seed={args.seed})")

        idx_dir = output_dir / args.dataset
        idx_dir.mkdir(parents=True, exist_ok=True)
        if selected_idx is not None:
            np.save(idx_dir / "selected_trial_idx.npy", selected_idx)
        np.save(idx_dir / "trial_labels.npy", np.array(trial_labels))

        for band_name in bands_to_run:
            fmin, fmax = FREQ_BANDS[band_name]
            print(f"\nComputing RDMs for band: {band_name} ({fmin}-{fmax} Hz)")

            if band_name == "broadband":
                data = epochs_data
            else:
                print("  Filtering...")
                data = bandpass_filter(epochs_data, sfreq, fmin, fmax)

            rdms, window_centers = compute_all_rdms(
                data, times, sfreq,
                tmin=args.tmin, tmax=args.tmax,
                window_ms=args.window_ms, step_ms=args.step_ms,
                region_mask=region_mask,
            )
            print(f"  Computed {len(rdms)} RDMs, shape: {rdms.shape}")

            band_dir = (output_dir / args.dataset / band_name
                        if args.region == "all_channels"
                        else output_dir / args.dataset / args.region / band_name)
            band_dir.mkdir(parents=True, exist_ok=True)
            np.save(band_dir / "rdms.npy", rdms)
            np.save(band_dir / "window_centers.npy", window_centers)

            metadata = {
                "dataset": args.dataset,
                "band": band_name,
                "region": args.region,
                "n_channels_used": int(region_mask.sum()) if region_mask is not None else int(epochs_data.shape[1]),
                "fmin": fmin, "fmax": fmax,
                "n_trials": epochs_data.shape[0],
                "n_trials_original": n_trials,
                "subsampled": selected_idx is not None,
                "max_trials": args.max_trials,
                "seed": args.seed,
                "n_windows": len(rdms),
                "window_ms": args.window_ms, "step_ms": args.step_ms,
                "tmin": args.tmin, "tmax": args.tmax,
                "sfreq": sfreq,
                "trial_labels": trial_labels[:20],
            }
            with open(band_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"  Saved to {band_dir}")

    print("\nAll RDMs computed and saved.")


if __name__ == "__main__":
    main()
