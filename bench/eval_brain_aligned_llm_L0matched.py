"""Capstone v3: brain-aligned GPT-2 fine-tune with MATCHED layer
(L0 target CID + L0 auxiliary loss) vs vanilla GPT-2 L0 on bench."""
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


def extract_layer(model, tok, words, layer_idx=0):
    out = np.zeros((len(words), 768), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i, w in enumerate(words):
            ids = tok(w, return_tensors="pt").input_ids
            if ids.shape[1] == 0: continue
            res = model.transformer(ids, output_hidden_states=True)
            out[i] = res.hidden_states[layer_idx][0, -1].numpy()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(42); torch.manual_seed(42)
    LAYER = 0  # matched

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    info = json.loads(
        (PROJ / "data/preprocessed/meg_masc/metadata.json").read_text())
    t_mask_full = np.linspace(info["epoch_tmin"], info["epoch_tmax"],
                              info["n_times"])
    t_mask = (t_mask_full >= 0.0) & (t_mask_full <= 0.8)  # post-onset

    n_train_sample = 1000
    n_eval_sample = 1500
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

    E_train = np.asarray(epochs[idx_train], dtype=np.float32)
    meg_feat_train = np.nan_to_num(
        E_train[:, :, t_mask].reshape(n_train_sample, -1)
    ).astype(np.float32)
    mu = meg_feat_train.mean(0, keepdims=True)
    sd = meg_feat_train.std(0, keepdims=True) + 1e-6
    meg_feat_train = (meg_feat_train - mu) / sd

    # Same featurization for eval (time-resolved, post-onset)
    E_eval = np.asarray(epochs[idx_eval], dtype=np.float32)
    meg_pat_eval_headline = np.nan_to_num(
        E_eval[:, :, (t_mask_full >= 0.65) & (t_mask_full <= 0.85)
              ].mean(axis=2))
    n_ch = E_eval.shape[1]
    ch = np.arange(n_ch // 4, 3 * n_ch // 4)
    meg_pat_eval_temp = meg_pat_eval_headline[:, ch]

    # Load L0-target CID decoder (v2 architecture: 4096, 2048, 1024)
    cid_decoder = CIDModel(n_meg_features=meg_feat_train.shape[1],
                            n_llm_dim=768,
                            hidden=(4096, 2048, 1024), dropout=0.0)
    cid_path = OUT / "cid_v2_model_L0_targets.pt"
    cid_decoder.load_state_dict(torch.load(cid_path, weights_only=True))
    for p in cid_decoder.parameters(): p.requires_grad_(False)
    cid_decoder.eval()
    print(f"[load] L0-target CID from {cid_path.name}", flush=True)

    # Vanilla GPT-2 L0 alignment
    print(f"[eval] vanilla GPT-2 layer {LAYER}", flush=True)
    model_van = GPT2LMHeadModel.from_pretrained("gpt2")
    emb_van = extract_layer(model_van, tok, words_eval, layer_idx=LAYER)
    z_van = headline_z(meg_pat_eval_temp, emb_van)
    print(f"  vanilla z(rho) = {z_van*1000:+.3f}", flush=True)

    # Brain-aligned fine-tune (MATCHED L0)
    print(f"[finetune] brain-aligned GPT-2, L{LAYER} target", flush=True)
    model_ft = GPT2LMHeadModel.from_pretrained("gpt2")
    opt = torch.optim.AdamW(model_ft.parameters(), lr=5e-5)
    cid_target_train = cid_decoder(torch.from_numpy(meg_feat_train).float())

    captured = {}
    def hook_fn(_module, _inp, out):
        captured["h"] = out[0] if isinstance(out, tuple) else out
    handle = model_ft.transformer.h[LAYER].register_forward_hook(hook_fn)

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
                h = captured["h"][0, -1]
                cos_sim = F.cosine_similarity(
                    h.unsqueeze(0),
                    cid_target_train[bi].unsqueeze(0),
                    dim=-1).squeeze()
                cid_losses.append(1.0 - cos_sim)
            if not losses: continue
            total = torch.stack(losses).mean() + \
                    0.5 * torch.stack(cid_losses).mean()
            total.backward(); opt.step()
        print(f"  ep {ep} done", flush=True)
    handle.remove()

    emb_ft = extract_layer(model_ft, tok, words_eval, layer_idx=LAYER)
    z_ft = headline_z(meg_pat_eval_temp, emb_ft)
    print(f"  brain-aligned z(rho) = {z_ft*1000:+.3f}", flush=True)
    delta = z_ft - z_van
    print(f"\n=== Brain-aligned vs vanilla (L0 matched) ===")
    print(f"  vanilla        z×10⁻³ = {z_van*1000:+.3f}")
    print(f"  brain-aligned  z×10⁻³ = {z_ft*1000:+.3f}")
    print(f"  Δz = {delta*1000:+.3f}")
    print(f"  relative: {100*delta/abs(z_van+1e-9):+.1f}%")

    pd.DataFrame([{"vanilla_z": z_van, "brain_aligned_z": z_ft,
                    "delta": delta, "layer": LAYER}]).to_csv(
        OUT / "brain_aligned_vs_vanilla_L0matched.tsv",
        sep="\t", index=False)


if __name__ == "__main__":
    main()
