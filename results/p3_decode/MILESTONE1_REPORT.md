# P3 Milestone 1 — Contrastive MEG→GPT-2 Decodability Probe (LibriBrain)

**Date:** 2026-06-30
**Cluster job:** `12117459` (word-level), `12117976` (3-s segment) on <cluster> `<gpu-partition>`
**Env:** `<HPC_SCRATCH>/envs/<env>/bin/python` (torch 2.6, CUDA)

---

## 1. Objective (from `P3_NEW_EXPERIMENT_DESIGN.md`, Design 1 / First Milestone)

Build the field-standard contrastive MEG→embedding retrieval decoder (Défossez/
Benchetrit architecture) and test the **gate**: does GPT-2 top-10 word-retrieval on a
held-out LibriBrain session clear **≈2–3× chance**? A pass proves there is decodable
signal that the naive ridge decoder (R² = −0.004) missed, and keeps P3 (the
decoding-direction test of P1's causal>masked asymmetry) alive.

## 2. What was built (paths, on <cluster> under `$R = <PROJECT>`)

| File | Role |
|---|---|
| `code/paper4_nn_revision/p3_extract_contextual_gpt2.py` | Re-extract **contextual** per-occurrence GPT-2 hidden states (layers 4/6/8/10/12) at each word's last sub-token, aligned to the 177,127 word-epochs. |
| `code/paper4_nn_revision/p3_train_infonce.py` | Conv MEG frontend (306-sensor 1×1 spatial mix + dilated temporal ResBlocks + time-pool → 768) + GPT-2 head + symmetric InfoNCE; **word-level** retrieval eval. |
| `code/paper4_nn_revision/p3_segment_retrieval.py` | Kill-criterion fallback: **3-s segment** retrieval (mean-pooled word-epochs & targets), same decoder. |
| `code/paper4_nn_revision/run_p3_milestone1.sbatch` | Driver: extraction → word-level training. |
| `code/paper4_nn_revision/run_p3_segment.sbatch` | 3-s segment run. |
| `results/p3_decode/gpt2_contextual.npy` | (177127, 5, 768) fp16 contextual hiddens; 177121/177127 filled. |
| `results/p3_decode/result_gpt2_L6.json` | Word-level gate result. |

**Critical fix — the cache was type-level, not contextual.** The design flagged that
`results/path1_libribrain/gpt2/hiddens.npy` is type-level. We confirmed the same for
`data/libribrain_full/llm/gpt2/{layer_12.npy,_emb_gpt2.pt}`: three different "the"
occurrences had cosine **1.0000** (byte-identical). After re-extraction, three "the"
occurrences have cosine **0.82 / 0.73 / 0.67** — genuinely contextual. This was the
single most important correctness fix (a contrastive decoder over identical repeated
targets cannot learn).

**Data / splits.** LibriBrain single subject, 12 sessions × {Sherlock1,2,3}. Brain side:
306-ch word-locked epochs, post-onset window **0–0.8 s** (samples 50:250, decimated ×2 →
100 timepoints). Train = sessions 1–11 (subsampled to 85,496 after dropping label-less
rows); held-out **test = session 12 / Sherlock3, N = 9,495 candidates**; 4,500 val from
training sessions. LLM side: contextual **GPT-2 layer 6** (768-d). Loss: symmetric
InfoNCE (CLIP), learnable temperature, AdamW, cosine LR, 40 epochs, best-val checkpoint.

## 3. GATE RESULT — word-level, held-out session 12 / Sherlock3 (N = 9,495)

| Metric | Decoder | Chance | **× chance** |
|---|---|---|---|
| top-1  | 0.032% | 0.0105% | **3.0×** |
| top-5  | 0.337% | 0.0527% | **6.4×** |
| top-10 | **0.642%** | 0.105% | **6.1×** |
| median rank | **2,143** / 9,495 | 4,747 (chance) | — |

**Shuffled-target control** (targets permuted before retrieval): top-10 = 0.074% = **0.7×
chance** (collapses to chance) → the 6.1× is not an artifact of the metric or pool.

**Training dynamics (important, honest).** For the first ~28 epochs the training loss fell
(6.04 → 2.0) while val retrieval sat at chance — this *looked* like pure overfitting. The
generalizing signal only emerged in the **late, low-LR epochs**: val top-10 went x2.3 (ep30)
→ x5.2 (ep36) → **x7.5 (ep38)**. The best-val checkpoint (val top-10 = 1.67%, 7.5× val
chance) produced the test numbers above. Absolute accuracy is low (top-10 = 0.64%), as
expected for **single-word, single-subject** MEG (cf. this project's own word-RDM≈0 finding;
Défossez's 72.5% was 3-s *segments*, and d'Ascoli decoded individual words only with 723
subjects) — but it is **robustly and significantly above chance**.

## 4. Verdict

**GATE CLEARED.** GPT-2 top-10 = **6.1× chance** (top-1 = 3.0×, top-5 = 6.4×), median rank
well below the chance midpoint (2,143 vs 4,747), and the shuffled control collapses to
chance. There **is** decodable MEG→LLM signal that the naive ridge (R² = −0.004) missed —
exactly the design's prediction that the failure was regression+RSA-features+averaged-
targets+R², not an absence of signal. The contextual-target fix + conv frontend + InfoNCE
recovers it.

**→ P3 is ALIVE.** Next step (Design 1 core experiment): swap the target family — matched
BERT/RoBERTa (masked), T5 (enc-dec), Pythia-70m/160m/410m (causal) — with identical layer-
fraction / projected-dim / splits / compute, and test the **causal-vs-masked decodability
contrast** (paired across matched layers, permutation test), plus the decodability-vs-layer
curve. The zoo is already cached for P1.

## 5. 3-s segment run (Défossez-comparable, robustness) — job 12117976 ✓

Higher-SNR 3-s-segment regime (mean-pooled word-epochs & mean-pooled contextual targets
within non-overlapping 3-s bins per recording; 18,306 train / 2,000 val / **1,229 test**
segments). Same decoder, 60 epochs. `result_gpt2_L6_seg3s.json`.

| Metric | Decoder | Chance | **× chance** |
|---|---|---|---|
| top-1  | 0.732% | 0.081% | **9.0×** |
| top-5  | 2.604% | 0.407% | **6.4×** |
| top-10 | **4.801%** | 0.814% | **5.9×** |
| median rank | **366** / 1,229 | 614 (chance) | — |

**Shuffled-target control:** top-10 = 0.65% = **0.8× chance** (collapses). Same
late-emergence dynamics (val top-10 x2.9 @ ep50 → **x6.0 @ ep59**). The segment regime
gives **~7.5× higher absolute top-10 (4.8% vs 0.64%)** and a stronger top-1 (9.0× chance)
than word-level — consistent with Défossez's finding that segment SNR ≫ single-word SNR.
Both regimes independently clear the gate.

## 6. Honest caveats

- **Repeated stimulus:** the three Sherlock stories recur across sessions, so held-out
  session 12 shares *text* (hence identical contextual targets) with training, though its
  *MEG* is unseen. This is the standard within-subject decoding regime (Défossez, d'Ascoli),
  but the causal-vs-masked comparison must keep splits identical across families so the
  contrast is fair.
- **Absolute accuracy is low** at word level; the causal-vs-masked contrast will likely be
  cleaner in the segment regime and should be reported there too.
- **Segment target construction:** mean-pooling contextual GPT-2 word embeddings over a 3-s
  bin lowers inter-segment variance (segments regress toward a common mean), so val looked
  like chance until the last epochs — yet the held-out test still cleared 5.9× (top-1 9.0×).
  A dedicated sentence/segment embedding (à la Défossez's wav2vec2) would likely raise this
  further; not needed for the gate.
- **Signal emerges late / LR-annealed:** in both runs retrieval sat at chance for the first
  ~60% of training and only generalized in the low-LR tail. Future runs should use more
  epochs / patience and select on val, not assume early overfitting == no signal.
- **Not committed:** no git commit performed (per instructions; lead logs + commits).
