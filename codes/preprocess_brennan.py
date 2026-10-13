#!/usr/bin/env python
"""
Preprocess Brennan 2019 EEG (49 subjects × 61 EEG + VEOG + Aux at 500 Hz,
Alice chapter 1, 12-min audio split into 12 segments).

Pipeline:
  1. Read S{NN}.vhdr (BrainVision triplet)
  2. Drop VEOG/Aux, keep 61 EEG channels (renumbered 1..61)
  3. Apply easycapM10-acti61 montage from .sfp
  4. Bandpass 0.5–45 Hz, notch 60 Hz, re-reference to average
  5. Downsample 500 → 250 Hz (matches LibriBrain/MEG-MASC/MEG-SCANS)
  6. Read AliceChapterOne-EEG.csv → 2128 words with (Segment, onset_within_segment)
  7. Find Stimulus/1..12 triggers in raw → segment-onset absolute times
  8. Compute word-onset = segment_onset + word.onset_within_segment
  9. Epoch (-200, +800) ms on each word
 10. Save to data/preprocessed/brennan_2019/

Output:
  data/preprocessed/brennan_2019/
      epochs_data.npy             [N_total_words, 61, n_samples]
      trial_metadata.tsv          word, surprisal_ngram/rnn/cfg, etc.
      metadata.json               {sfreq, epoch_tmin, epoch_tmax, n_subjects}
      stimuli_words.txt           one word per line, Alice chap 1 in order

Usage:
  python codes/preprocess_brennan.py
  python codes/preprocess_brennan.py --subjects S01 S02   # subset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("<REPO_ROOT>")
RAW = Path("<HOME>/alice_eeg")
META = PROJ / "data/brennan_2019_meta"
OUT = PROJ / "data/preprocessed/brennan_2019"

EPOCH_TMIN = -0.2
EPOCH_TMAX = 0.8
SFREQ_TARGET = 250  # downsample target


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="*", default=None,
                    help="Subject IDs (e.g. S01 S02); default = all 49")
    p.add_argument("--max_subjects", type=int, default=49)
    args = p.parse_args()

    print("[load] mne", flush=True)
    import mne

    OUT.mkdir(parents=True, exist_ok=True)

    print(f"[load] alignment metadata from {META}", flush=True)
    align = pd.read_csv(META / "AliceChapterOne-EEG.csv")
    print(f"  {len(align)} words × {len(align.columns)} columns", flush=True)
    # Sentence index: cumsum of new sentences (Sentence column starts at 1
    # for each new sentence). Convert to a 0-indexed word_idx within a
    # sentence so we can group "sentence" later for LLM contexts.
    align["sentenceidx"] = (align["Position"] == 1).cumsum() - 1
    align["wordidx"] = align.groupby("sentenceidx").cumcount()

    # Subjects
    if args.subjects is not None:
        subjects = args.subjects
    else:
        vhdr_files = sorted(RAW.glob("S??.vhdr"))[: args.max_subjects]
        subjects = [v.stem for v in vhdr_files]
    print(f"[subjects] {len(subjects)} subjects: {subjects[:3]}…{subjects[-3:]}",
            flush=True)

    # Read electrode positions (.sfp): "<idx>\t<x>\t<y>\t<z>"
    sfp_path = META / "easycapM10-acti61_elec.sfp"
    sfp = pd.read_csv(sfp_path, sep="\t", header=None,
                       names=["label", "x", "y", "z"])
    # Channel indices in vhdr are 1..61, label them by position number
    chan_pos = {str(int(row["label"])): np.array([row["x"], row["y"], row["z"]])
                for _, row in sfp.iterrows()}
    print(f"  electrode positions: {len(chan_pos)} channels", flush=True)

    all_epochs = []
    all_meta_rows = []

    for sub in subjects:
        vhdr = RAW / f"{sub}.vhdr"
        if not vhdr.exists():
            print(f"  [skip] {sub}: missing {vhdr}", flush=True)
            continue
        print(f"\n[process] {sub}", flush=True)
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True,
                                                verbose="ERROR")
        except Exception as e:
            print(f"  [skip] {sub}: read error {e}", flush=True)
            continue

        # Keep only numbered channels '1'..'61' (drops VEOG, Aux*, mastoid refs, etc.)
        keep = [c for c in raw.ch_names if c.isdigit() and 1 <= int(c) <= 61]
        if len(keep) != 61:
            print(f"  [warn] {sub}: only {len(keep)} numbered channels (expected 61)",
                    flush=True)
        raw.pick_channels(keep, ordered=True)
        # Set channel positions from .sfp
        montage = mne.channels.make_dig_montage(
            ch_pos={ch: chan_pos[ch] / 100  # cm → m
                    for ch in raw.ch_names if ch in chan_pos},
            coord_frame="head",
        )
        raw.set_montage(montage, on_missing="warn", verbose="ERROR")

        # Filter
        raw.filter(0.5, 45.0, fir_design="firwin", verbose="ERROR")
        raw.notch_filter(60.0, verbose="ERROR")
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")

        # Downsample
        raw.resample(SFREQ_TARGET, npad="auto", verbose="ERROR")

        # Find segment triggers — handle both 'Stimulus/N' and 'Stimulus/S  N' formats
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        seg_onsets_s = {}  # segment_id (1..12) → onset time in seconds
        for stim_id in range(1, 13):
            for key_fmt in (f"Stimulus/{stim_id}",
                              f"Stimulus/S  {stim_id}",
                              f"Stimulus/S {stim_id}",
                              f"Stimulus/S{stim_id:2d}"):
                if key_fmt in event_id:
                    ev_id = event_id[key_fmt]
                    ev_idx = np.where(events[:, 2] == ev_id)[0]
                    if len(ev_idx) > 0:
                        seg_onsets_s[stim_id] = events[ev_idx[0], 0] / raw.info["sfreq"]
                    break
        print(f"  segment onsets found: {len(seg_onsets_s)}/12",
                flush=True)

        # Build word onsets
        word_events = []
        meta_rows = []
        for _, w in align.iterrows():
            seg = int(w["Segment"])
            if seg not in seg_onsets_s:
                continue
            t_abs = seg_onsets_s[seg] + float(w["onset"])
            samp = int(round(t_abs * raw.info["sfreq"]))
            if samp < int(0.3 * raw.info["sfreq"]):
                continue
            if samp >= len(raw.times) - int(1.0 * raw.info["sfreq"]):
                continue
            word_events.append([samp, 0, 1])
            meta_rows.append({
                "subject": sub,
                "session": sub,
                "task": "alice_ch1",
                "run": "01",
                "Word": w["Word"],
                "word_text": w["Word"],
                "Order": int(w["Order"]),
                "Segment": seg,
                "Position": int(w["Position"]),
                "Sentence": int(w["Sentence"]),
                "sentenceidx": int(w["sentenceidx"]),
                "wordidx": int(w["wordidx"]),
                "IsLexical": int(w["IsLexical"]) if pd.notna(w["IsLexical"]) else -1,
                "LogFreq": float(w["LogFreq"]) if pd.notna(w["LogFreq"]) else np.nan,
                "SndPower": float(w["SndPower"]) if pd.notna(w["SndPower"]) else np.nan,
                "Length": float(w["Length"]) if pd.notna(w["Length"]) else np.nan,
                "surprisal_ngram": float(w["NGRAM"])
                                    if pd.notna(w["NGRAM"]) else np.nan,
                "surprisal_rnn": float(w["RNN"])
                                  if pd.notna(w["RNN"]) else np.nan,
                "surprisal_cfg": float(w["CFG"])
                                  if pd.notna(w["CFG"]) else np.nan,
                "onset_in_segment": float(w["onset"]),
                "onset_abs": t_abs,
            })

        if not word_events:
            print(f"  [skip] {sub}: no valid words after epoching cuts",
                    flush=True)
            continue

        word_events = np.array(word_events, dtype=np.int64)
        epochs = mne.Epochs(raw, word_events, event_id={"word": 1},
                              tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
                              baseline=(EPOCH_TMIN, 0.0),
                              preload=True, verbose="ERROR")
        # Drop bad epochs (default rejection threshold)
        # epochs.drop_bad()
        data = epochs.get_data()  # [N, 61, T]
        print(f"  {sub}: {data.shape} epochs, sfreq={raw.info['sfreq']}",
                flush=True)
        all_epochs.append(data.astype(np.float32))
        all_meta_rows.extend(meta_rows[: len(data)])

    if not all_epochs:
        print("ERROR: no epochs to save", flush=True)
        return

    # Concatenate
    print("\n[save] concatenating across subjects", flush=True)
    epochs_data = np.concatenate(all_epochs, axis=0)
    print(f"  total: {epochs_data.shape}", flush=True)

    np.save(OUT / "epochs_data.npy", epochs_data)
    meta_df = pd.DataFrame(all_meta_rows)
    meta_df.to_csv(OUT / "trial_metadata.tsv", sep="\t", index=False)
    # Channel names: numbered '1'..'61' (or whatever the kept set was)
    ch_names = [str(i) for i in range(1, epochs_data.shape[1] + 1)]
    metadata = {
        "sfreq": SFREQ_TARGET,
        "epoch_tmin": EPOCH_TMIN,
        "epoch_tmax": EPOCH_TMAX,
        "n_subjects": len(set(meta_df["subject"])),
        "n_trials_total": len(meta_df),
        "n_epochs": int(epochs_data.shape[0]),
        "n_channels": int(epochs_data.shape[1]),
        "n_times": int(epochs_data.shape[2]),
        "n_samples_per_epoch": int(epochs_data.shape[2]),
        "ch_names": ch_names,
        "atlas": "easycap61",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Save stimuli words list (single subject's words; same across subjects)
    sub0 = meta_df[meta_df["subject"] == meta_df["subject"].iloc[0]]
    (OUT / "stimuli_words.txt").write_text(
        "\n".join(sub0["word_text"].fillna("").astype(str).tolist()))

    print(f"\n[saved] {OUT}", flush=True)
    print(f"  epochs_data.npy: {epochs_data.shape}", flush=True)
    print(f"  trial_metadata.tsv: {len(meta_df)} rows", flush=True)
    print(f"  metadata.json: {metadata}", flush=True)


if __name__ == "__main__":
    main()
