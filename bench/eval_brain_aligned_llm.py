"""Capstone: does the brain-aligned fine-tuned GPT-2 outperform vanilla
on the bench best-layer task?

Pipeline:
  1. Fine-tune GPT-2 with CID-auxiliary loss on MEG-MASC × 1000 trials
  2. Extract layer-5 activations from both vanilla + fine-tuned GPT-2
     on the same evaluation trials
  3. Compute streaming-RSA z(ρ) against MEG-MASC temporal_R × +768ms
     headline cell for both models
  4. Report Δz = z(fine-tuned) − z(vanilla)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, "<HOME>/MEG_paper4/causal_decoder")
from model import CIDModel


PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "results/causal_decoder"


def fisher_z(rho):
    return float(np.arctanh(np.clip(rho, -0.999, 0.999)))


def headline_z(meg_pat, llm_arr):
    if meg_pat.shape[0] < 100: return float("nan")
    rdm_m = np.nan_to_num(pdist(meg_pat, metric="correlation"), nan=0.0)
    rdm_l = np.nan_to_num(pdist(llm_arr, metric="correlation"), nan=0.0)
    if rdm_m.std() == 0 or rdm_l.std() == 0: return float("nan")
    return fisher_z(float(np.corrcoef(rankdata(rdm_m), rankdata(rdm_l))[0, 1]))


def extract_layer5(model, tok, words):
    """Get L5 activations for each word, last subtoken."""
    out = np.zeros((len(words), 768), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i, w in enumerate(words):
            ids = tok(w, return_tensors="pt").input_ids
            if ids.shape[1] == 0: continue
            res = model.transformer(ids, output_hidden_states=True)
            h5 = res.hidden_states[5][0, -1]
            out[i] = h5.numpy()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(42); torch.manual_seed(42)

    # Load
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    info = json.loads(
        (PROJ / "data/preprocessed/meg_masc/metadata.json").read_text())
    epoch_tmin = float(info["epoch_tmin"])
    epoch_tmax = float(info["epoch_tmax"])
    n_times = int(info["n_times"])
    times = np.linspace(epoch_tmin, epoch_tmax, n_times)
    t_mask = (times >= 0.65) & (times <= 0.85)  # MEG-MASC headline

    n_train_sample = 1000  # for fine-tuning
    n_eval_sample = 1500   # for benchmarking
    epochs = np.load(PROJ / "data/preprocessed/meg_masc/epochs_data.npy",
                     mmap_mode="r")
    rng = np.random.default_rng(42)
    idx_train = np.sort(rng.choice(epochs.shape[0], n_train_sample,
                                   replace=False))
    idx_eval = np.sort(rng.choice(epochs.shape[0], n_eval_sample,
                                  replace=False))

    meta = pd.read_csv(
        PROJ / "data/preprocessed/meg_masc/trial_metadata.tsv", sep="\t")
    word_col = "word" if "word" in meta.columns else "word_text"
    words_train = [str(meta.iloc[i][word_col]) for i in idx_train]
    words_eval = [str(meta.iloc[i][word_col]) for i in idx_eval]

    # MEG patterns
    E_train = np.asarray(epochs[idx_train], dtype=np.float32)
    meg_pat_train = np.nan_to_num(E_train[:, :, t_mask].mean(axis=2))
    E_eval = np.asarray(epochs[idx_eval], dtype=np.float32)
    meg_pat_eval = np.nan_to_num(E_eval[:, :, t_mask].mean(axis=2))

    # Use fewer channels for SVD speed: just temporal_R approximation
    # (middle 50% channels)
    n_ch = E_eval.shape[1]
    ch = np.arange(n_ch // 4, 3 * n_ch // 4)
    meg_pat_eval_temp = np.nan_to_num(
        E_eval[:, ch, :][:, :, t_mask].mean(axis=2))

    print(f"[setup] {n_train_sample} train, {n_eval_sample} eval trials",
          flush=True)

    # === Load frozen CID decoder ===
    cid_decoder = CIDModel(n_meg_features=208, n_llm_dim=768,
                            hidden=(2048, 1024), dropout=0.0)
    cid_path = OUT / "cid_model.pt"
    if cid_path.exists():
        cid_decoder.load_state_dict(torch.load(cid_path, weights_only=True))
    for p in cid_decoder.parameters():
        p.requires_grad_(False)
    cid_decoder.eval()

    # === Vanilla GPT-2 baseline ===
    print("[eval] vanilla GPT-2 layer 5", flush=True)
    model_van = GPT2LMHeadModel.from_pretrained("gpt2")
    emb_van = extract_layer5(model_van, tok, words_eval)
    z_van = headline_z(meg_pat_eval_temp, emb_van)
    print(f"  vanilla GPT-2 L5  z(rho) = {z_van*1000:+.3f}",
          flush=True)

    # === Brain-aligned fine-tune ===
    print("[finetune] brain-aligned GPT-2 fine-tune (5 epochs)", flush=True)
    model_ft = GPT2LMHeadModel.from_pretrained("gpt2")
    opt = torch.optim.AdamW(model_ft.parameters(), lr=5e-5)
    cid_target_train = cid_decoder(torch.from_numpy(meg_pat_train).float())

    captured = {}
    def hook_fn(_module, _inp, out):
        captured["h5"] = out[0] if isinstance(out, tuple) else out
    handle = model_ft.transformer.h[5].register_forward_hook(hook_fn)

    model_ft.train()
    for ep in range(5):
        for i in range(0, len(words_train), 8):
            batch = list(range(i, min(i+8, len(words_train))))
            opt.zero_grad()
            losses = []; cid_losses = []
            for bi in batch:
                w = words_train[bi]
                ids = tok(w, return_tensors="pt").input_ids
                if ids.shape[1] < 2: continue
                out = model_ft(ids, labels=ids)
                losses.append(out.loss)
                h5 = captured["h5"][0, -1]
                cos_sim = F.cosine_similarity(
                    h5.unsqueeze(0),
                    cid_target_train[bi].unsqueeze(0),
                    dim=-1).squeeze()
                cid_losses.append(1.0 - cos_sim)
            if not losses: continue
            total = torch.stack(losses).mean() + \
                    0.5 * torch.stack(cid_losses).mean()
            total.backward()
            opt.step()
        print(f"  ep {ep} done", flush=True)
    handle.remove()

    # === Eval brain-aligned model ===
    print("[eval] brain-aligned GPT-2 layer 5", flush=True)
    emb_ft = extract_layer5(model_ft, tok, words_eval)
    z_ft = headline_z(meg_pat_eval_temp, emb_ft)
    print(f"  brain-aligned GPT-2 L5  z(rho) = {z_ft*1000:+.3f}",
          flush=True)

    delta = z_ft - z_van
    print(f"\n=== Brain-aligned GPT-2 vs vanilla on MEG-MASC bench ===")
    print(f"  vanilla        z(rho) × 10⁻³ = {z_van*1000:+.3f}")
    print(f"  brain-aligned  z(rho) × 10⁻³ = {z_ft*1000:+.3f}")
    print(f"  Δz (brain-aligned − vanilla) = {delta*1000:+.3f}")
    print(f"  relative improvement: {100*delta/abs(z_van+1e-9):+.1f}%")
    pd.DataFrame([{"vanilla_z": z_van, "brain_aligned_z": z_ft,
                    "delta": delta}]).to_csv(
        OUT / "brain_aligned_vs_vanilla.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
