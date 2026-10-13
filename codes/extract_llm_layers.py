"""
Paper 4 - OscilAlign
Step 2: Extract context-aware LLM layer-wise embeddings for each word/trial.

V2 improvement: instead of feeding each word independently to the LLM,
reconstruct sentence context from events TSV (using sentenceidx) and extract
the hidden state at the target word's position within the full sentence.
This follows the standard approach of Caucheteux & King (2022).

Two modes:
  --context_mode word    : V1 mode (each word independently, backward compat)
  --context_mode sentence: V2 mode (sentence context, recommended)

Models (V2): GPT-2, GPT-2-medium, BERT-base-uncased

Usage:
    # Context-aware mode (recommended)
    python extract_llm_layers.py \
        --stimuli_file data/preprocessed/libribrain/stimuli_words.txt \
        --events_dir data/libribrain \
        --output_dir data/preprocessed/llm_embeddings \
        --context_mode sentence \
        --models gpt2 gpt2-medium bert-base-uncased

    # Legacy word-by-word mode
    python extract_llm_layers.py \
        --stimuli_file data/preprocessed/libribrain/stimuli_words.txt \
        --output_dir data/preprocessed/llm_embeddings \
        --context_mode word \
        --models gpt2

Requirements: transformers>=4.30, torch (CPU-only), numpy, pandas
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_words(stimuli_file: str) -> list[str]:
    """Load one word per line from a stimulus file."""
    with open(stimuli_file) as f:
        words = [line.strip() for line in f if line.strip()]
    return words


def _safe_int(value):
    """Prefer numeric sorting when identifiers are integers."""
    if value is None:
        return ""
    text = str(value)
    return int(text) if text.isdigit() else text


def _parse_events_file_info(events_file: Path) -> dict:
    """Parse task/session/run identifiers from an events filename."""
    info = {"task": "", "session": "", "run": ""}
    for part in events_file.stem.split("_"):
        if part.startswith("ses-"):
            info["session"] = part[4:]
        elif part.startswith("task-"):
            info["task"] = part[5:]
        elif part.startswith("run-"):
            info["run"] = part[4:]
    return info


def _events_sort_key(events_file: Path) -> tuple:
    info = _parse_events_file_info(events_file)
    return (
        info["task"],
        _safe_int(info["session"]),
        _safe_int(info["run"]),
        events_file.name,
    )


def _load_word_events_df(events_file: Path) -> pd.DataFrame:
    """Load word events with stable row ids and optional sentence metadata."""
    df = pd.read_csv(events_file, sep="\t")
    if "kind" not in df.columns:
        return pd.DataFrame()

    words_df = df[df["kind"] == "word"].copy()
    if words_df.empty:
        return pd.DataFrame()

    words_df["event_row_index"] = words_df.index.astype(int)

    word_col = None
    for col in ["segment", "word", "text", "value", "trial_type"]:
        if col in words_df.columns:
            word_col = col
            break
    if word_col is None:
        words_df["word_text"] = [f"word_{i}" for i in range(len(words_df))]
    else:
        words_df["word_text"] = words_df[word_col].astype(str)

    words_df["sentenceidx"] = pd.to_numeric(
        words_df["sentenceidx"], errors="coerce"
    ) if "sentenceidx" in words_df.columns else np.nan
    words_df["wordidx"] = pd.to_numeric(
        words_df["wordidx"], errors="coerce"
    ) if "wordidx" in words_df.columns else np.nan

    return words_df


# --------------------------------------------------------------------------- #
# Sentence context reconstruction from events TSV
# --------------------------------------------------------------------------- #

def reconstruct_sentences_from_events(data_dir: Path) -> list[dict]:
    """
    Reconstruct sentence contexts from LibriBrain events TSV files.

    Groups word events by (task, session, sentenceidx) to build sentences.
    Returns a list of dicts, one per word event in presentation order:
        {
            "word": str,           # target word
            "sentence": str,       # full sentence context
            "word_position": int,  # 0-indexed position of target word in sentence
            "sentence_words": list[str],  # all words in the sentence
            "task": str,
            "session": str,
            "sentenceidx": float,
        }
    """
    all_word_records = []

    for task_dir in sorted(data_dir.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        events_dir = task_dir / "derivatives" / "events"
        if not events_dir.exists():
            continue

        for events_file in sorted(events_dir.glob("*_events.tsv"), key=_events_sort_key):
            info = _parse_events_file_info(events_file)
            words_df = _load_word_events_df(events_file)
            if words_df.empty:
                continue

            words_df["sent_idx"] = words_df["sentenceidx"]
            words_df["word_idx"] = words_df["wordidx"]

            # Group by sentence
            for sent_idx, sent_group in words_df.groupby("sent_idx", sort=True):
                if pd.isna(sent_idx):
                    continue
                sent_sorted = sent_group.sort_values("word_idx")
                sentence_words = sent_sorted["word_text"].tolist()
                sentence_str = " ".join(sentence_words)

                for pos, (_, row) in enumerate(sent_sorted.iterrows()):
                    all_word_records.append({
                        "word": row["word_text"],
                        "sentence": sentence_str,
                        "word_position": pos,
                        "sentence_words": sentence_words,
                        "task": info.get("task", ""),
                        "session": info.get("session", ""),
                        "sentenceidx": float(sent_idx),
                    })

    return all_word_records


def build_sentence_contexts_from_trial_metadata(
    trial_metadata_file: Path,
) -> list[dict]:
    """
    Build sentence contexts for exactly the retained trials in preprocessing order.

    This avoids mismatch when preprocessing used only a subset of sessions/tasks.
    The trial metadata stores the original events row index, which lets us
    reconstruct the full sentence from the source events file while preserving
    exact epoch order.
    """
    trial_df = pd.read_csv(trial_metadata_file, sep="\t")
    if trial_df.empty:
        return []

    # Preferred path for datasets that already store per-trial context explicitly,
    # e.g. MEG-MASC after Paper4-specific preprocessing.
    direct_cols = {"word_text", "sentence", "word_position"}
    if direct_cols.issubset(trial_df.columns):
        contexts = []
        for _, row in trial_df.iterrows():
            word = str(row["word_text"])
            sentence = str(row["sentence"]) if pd.notna(row["sentence"]) else word
            if "sentence_words" in trial_df.columns and pd.notna(row.get("sentence_words")):
                sentence_words = str(row["sentence_words"]).split()
            else:
                sentence_words = sentence.split()
            if not sentence_words:
                sentence_words = [word]
            try:
                word_position = int(row["word_position"])
            except Exception:
                word_position = 0
            word_position = max(0, min(word_position, len(sentence_words) - 1))
            contexts.append({
                "word": word,
                "sentence": sentence,
                "word_position": word_position,
                "sentence_words": sentence_words,
                "task": str(row["task"]) if "task" in trial_df.columns and pd.notna(row["task"]) else "",
                "session": str(row["session"]) if "session" in trial_df.columns and pd.notna(row["session"]) else "",
                "story": str(row["story"]) if "story" in trial_df.columns and pd.notna(row["story"]) else "",
                "sequence_id": row["sequence_id"] if "sequence_id" in trial_df.columns else np.nan,
            })
        return contexts

    required_cols = {"word_text", "source_events_path", "event_row_index"}
    if not required_cols.issubset(trial_df.columns):
        print(
            "  WARNING: trial metadata missing required columns; "
            "falling back to stimuli/events matching"
        )
        return []

    context_lookup = {}
    unique_event_files = [
        Path(path_str)
        for path_str in pd.unique(trial_df["source_events_path"].dropna())
    ]

    for events_file in unique_event_files:
        words_df = _load_word_events_df(events_file)
        if words_df.empty:
            continue

        if words_df["sentenceidx"].notna().any():
            grouped = words_df.groupby("sentenceidx", sort=True)
            for sent_idx, sent_group in grouped:
                sent_sorted = sent_group.sort_values(
                    by=["wordidx", "event_row_index"], na_position="last"
                )
                sentence_words = sent_sorted["word_text"].tolist()
                sentence_str = " ".join(sentence_words)

                for pos, (_, row) in enumerate(sent_sorted.iterrows()):
                    context_lookup[(str(events_file), int(row["event_row_index"]))] = {
                        "word": row["word_text"],
                        "sentence": sentence_str,
                        "word_position": pos,
                        "sentence_words": sentence_words,
                        "task": _parse_events_file_info(events_file)["task"],
                        "session": _parse_events_file_info(events_file)["session"],
                        "sentenceidx": float(sent_idx),
                    }
        else:
            for _, row in words_df.iterrows():
                context_lookup[(str(events_file), int(row["event_row_index"]))] = {
                    "word": row["word_text"],
                    "sentence": row["word_text"],
                    "word_position": 0,
                    "sentence_words": [row["word_text"]],
                    "task": _parse_events_file_info(events_file)["task"],
                    "session": _parse_events_file_info(events_file)["session"],
                    "sentenceidx": np.nan,
                }

    contexts = []
    missing = 0
    mismatch = 0
    for _, row in trial_df.iterrows():
        key = (str(row["source_events_path"]), int(row["event_row_index"]))
        ctx = context_lookup.get(key)
        word = str(row["word_text"])
        if ctx is None:
            missing += 1
            contexts.append({
                "word": word,
                "sentence": word,
                "word_position": 0,
                "sentence_words": [word],
            })
            continue

        if ctx["word"] != word:
            mismatch += 1
            ctx = {**ctx, "word": word}
        contexts.append(ctx)

    if missing > 0:
        print(f"  WARNING: missing sentence context for {missing} trials")
    if mismatch > 0:
        print(f"  WARNING: word mismatch in {mismatch} trials; metadata word was kept")

    return contexts


def build_sentence_contexts(
    words: list[str],
    data_dir: Path,
    min_word_len: int = 2,
) -> list[dict]:
    """
    Build sentence contexts matching the preprocessed word list.

    The preprocessed stimuli_words.txt contains words in the same order
    as they were extracted by preprocess_libribrain_paper4.py. We reconstruct
    contexts from events and match by word text + order.

    Returns list of dicts with sentence context for each word in the stimuli.
    """
    all_records = reconstruct_sentences_from_events(data_dir)

    # Filter by min_word_len (same as preprocessing)
    filtered = [r for r in all_records if len(r["word"]) >= min_word_len]

    if len(filtered) != len(words):
        print(f"  WARNING: sentence records ({len(filtered)}) != stimuli words ({len(words)})")
        print(f"  First 5 records: {[r['word'] for r in filtered[:5]]}")
        print(f"  First 5 stimuli: {words[:5]}")
        # Try to match by truncating to the shorter length
        n = min(len(filtered), len(words))
        # Verify alignment
        mismatches = sum(1 for i in range(min(n, 100))
                        if filtered[i]["word"] != words[i])
        if mismatches > 10:
            print(f"  WARNING: {mismatches}/100 mismatches in first 100 words!")
            print(f"  Falling back to word-only mode")
            return [{"word": w, "sentence": w, "word_position": 0,
                     "sentence_words": [w]} for w in words]
        filtered = filtered[:n]

    return filtered


# --------------------------------------------------------------------------- #
# Context-aware extraction: GPT-2 family
# --------------------------------------------------------------------------- #

def extract_gpt2_contextual(
    contexts: list[dict],
    model_name: str = "gpt2",
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
) -> dict[str, np.ndarray]:
    """
    Extract per-layer hidden states for each word using sentence context.

    For GPT-2 (autoregressive): input the sentence up to and including the
    target word, take the hidden state at the last token of the target word.
    """
    from transformers import AutoTokenizer, GPT2Model

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    model.to(device)

    n_layers = model.config.n_layer
    layer_names = [f"layer_{i}" for i in range(n_layers + 1)]
    all_layer_vecs = {name: [] for name in layer_names}

    n_words = len(contexts)

    with torch.no_grad():
        for i in range(0, n_words, batch_size):
            batch_contexts = contexts[i : i + batch_size]

            # For GPT-2: use prefix up to target word
            batch_texts = []
            target_words = []
            for ctx in batch_contexts:
                # Build prefix: all words up to and including target
                pos = ctx["word_position"]
                prefix_words = ctx["sentence_words"][:pos + 1]
                prefix = " ".join(prefix_words)
                batch_texts.append(prefix)
                target_words.append(ctx["word"])

            inputs = tokenizer(
                batch_texts, return_tensors="pt",
                padding=True, truncation=True, max_length=512,
            ).to(device)
            attention_mask = inputs["attention_mask"]
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

            # For each sample, get the last non-padding token's hidden state
            # (which corresponds to the last token of the target word)
            seq_lens = attention_mask.sum(dim=1) - 1  # last valid index

            for layer_idx, layer_name in enumerate(layer_names):
                hs = hidden_states[layer_idx]  # [B, seq_len, dim]
                vecs = []
                for b in range(len(batch_texts)):
                    last_idx = seq_lens[b].item()
                    vecs.append(hs[b, last_idx].cpu().numpy())
                all_layer_vecs[layer_name].append(np.stack(vecs))

            if (i // batch_size) % 10 == 0:
                print(f"    {min(i + batch_size, n_words)}/{n_words} words processed")

    return {name: np.concatenate(vecs, axis=0) for name, vecs in all_layer_vecs.items()}


# --------------------------------------------------------------------------- #
# Context-aware extraction: BERT family
# --------------------------------------------------------------------------- #

def extract_bert_contextual(
    contexts: list[dict],
    model_name: str = "bert-base-uncased",
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
) -> dict[str, np.ndarray]:
    """
    Extract per-layer hidden states for each word using sentence context.

    For BERT (bidirectional): input the full sentence, find the target word's
    token(s), and average their hidden states.
    """
    from transformers import AutoModel, AutoTokenizer

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    model.to(device)

    config = model.config
    n_layers = getattr(config, "num_hidden_layers", 12)
    layer_names = [f"layer_{i}" for i in range(n_layers + 1)]
    all_layer_vecs = {name: [] for name in layer_names}

    n_words = len(contexts)

    with torch.no_grad():
        for i in range(0, n_words, batch_size):
            batch_contexts = contexts[i : i + batch_size]

            batch_sentences = [ctx["sentence"] for ctx in batch_contexts]
            target_words = [ctx["word"] for ctx in batch_contexts]

            inputs = tokenizer(
                batch_sentences, return_tensors="pt",
                padding=True, truncation=True, max_length=512,
            ).to(device)
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

            # Find target word token indices for each sample
            for layer_idx, layer_name in enumerate(layer_names):
                hs = hidden_states[layer_idx]  # [B, seq_len, dim]
                vecs = []
                for b in range(len(batch_sentences)):
                    # Tokenize the target word alone to find its tokens
                    word_tokens = tokenizer.encode(
                        target_words[b], add_special_tokens=False
                    )
                    input_ids = inputs["input_ids"][b].tolist()

                    # Find the word tokens in the input sequence
                    target_indices = _find_sublist(input_ids, word_tokens)
                    if target_indices:
                        # Average hidden states over target word tokens
                        vec = hs[b, target_indices].mean(dim=0).cpu().numpy()
                    else:
                        # Fallback: use [CLS] token
                        vec = hs[b, 0].cpu().numpy()
                    vecs.append(vec)
                all_layer_vecs[layer_name].append(np.stack(vecs))

            if (i // batch_size) % 10 == 0:
                print(f"    {min(i + batch_size, n_words)}/{n_words} words processed")

    return {name: np.concatenate(vecs, axis=0) for name, vecs in all_layer_vecs.items()}


def _find_sublist(lst: list, sublist: list) -> list[int]:
    """Find indices of sublist within lst. Returns indices or empty list."""
    n = len(sublist)
    for start in range(len(lst) - n + 1):
        if lst[start : start + n] == sublist:
            return list(range(start, start + n))
    return []


def _init_layer_memmaps(
    output_dir: Path,
    model_name: str,
    layer_names: list[str],
    n_words: int,
    hidden_dim: int,
) -> tuple[Path, dict[str, np.memmap]]:
    """Create one .npy memmap per layer for streaming writes."""
    model_dir = output_dir / model_name.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    memmaps = {}
    for layer_name in layer_names:
        memmaps[layer_name] = open_memmap(
            model_dir / f"{layer_name}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_words, hidden_dim),
        )
    return model_dir, memmaps


def _finalize_stream_save(
    model_dir: Path,
    model_name: str,
    layer_names: list[str],
    n_words: int,
    hidden_dim: int,
    context_mode: str,
) -> None:
    """Write metadata for a streamed embedding extraction."""
    for mem in layer_names:
        pass
    metadata = {
        "model_name": model_name,
        "n_words": n_words,
        "context_mode": context_mode,
        "layers": [
            {"name": layer_name, "shape": [n_words, hidden_dim]}
            for layer_name in layer_names
        ],
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved {len(layer_names)} layers to {model_dir}")


def extract_gpt2_contextual_to_disk(
    contexts: list[dict],
    output_dir: Path,
    model_name: str = "gpt2",
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Stream GPT-2 contextual embeddings directly to disk."""
    from transformers import AutoTokenizer, GPT2Model

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    model.to(device)

    n_layers = model.config.n_layer
    hidden_dim = model.config.n_embd
    layer_names = [f"layer_{i}" for i in range(n_layers + 1)]
    n_words = len(contexts)
    model_dir, memmaps = _init_layer_memmaps(
        output_dir, model_name, layer_names, n_words, hidden_dim
    )

    with torch.no_grad():
        for i in range(0, n_words, batch_size):
            batch_contexts = contexts[i : i + batch_size]
            batch_texts = []
            for ctx in batch_contexts:
                pos = ctx["word_position"]
                prefix_words = ctx["sentence_words"][:pos + 1]
                batch_texts.append(" ".join(prefix_words))

            inputs = tokenizer(
                batch_texts, return_tensors="pt",
                padding=True, truncation=True, max_length=512,
            ).to(device)
            attention_mask = inputs["attention_mask"]
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states
            seq_lens = attention_mask.sum(dim=1) - 1

            for layer_idx, layer_name in enumerate(layer_names):
                hs = hidden_states[layer_idx]
                vecs = []
                for b in range(len(batch_texts)):
                    last_idx = seq_lens[b].item()
                    vecs.append(hs[b, last_idx].cpu().numpy())
                memmaps[layer_name][i : i + len(batch_texts)] = np.stack(vecs).astype(np.float32)

            if (i // batch_size) % 10 == 0:
                print(f"    {min(i + batch_size, n_words)}/{n_words} words processed")

    for mem in memmaps.values():
        mem.flush()
    _finalize_stream_save(model_dir, model_name, layer_names, n_words, hidden_dim, "sentence")


def extract_bert_contextual_to_disk(
    contexts: list[dict],
    output_dir: Path,
    model_name: str = "bert-base-uncased",
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Stream BERT-family contextual embeddings directly to disk."""
    from transformers import AutoModel, AutoTokenizer

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    model.to(device)

    config = model.config
    n_layers = getattr(config, "num_hidden_layers", 12)
    hidden_dim = getattr(config, "hidden_size", None)
    if hidden_dim is None:
        raise ValueError(f"Could not infer hidden size for {model_name}")

    layer_names = [f"layer_{i}" for i in range(n_layers + 1)]
    n_words = len(contexts)
    model_dir, memmaps = _init_layer_memmaps(
        output_dir, model_name, layer_names, n_words, hidden_dim
    )

    with torch.no_grad():
        for i in range(0, n_words, batch_size):
            batch_contexts = contexts[i : i + batch_size]
            batch_sentences = [ctx["sentence"] for ctx in batch_contexts]
            target_words = [ctx["word"] for ctx in batch_contexts]

            inputs = tokenizer(
                batch_sentences, return_tensors="pt",
                padding=True, truncation=True, max_length=512,
            ).to(device)
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

            for layer_idx, layer_name in enumerate(layer_names):
                hs = hidden_states[layer_idx]
                vecs = []
                for b in range(len(batch_sentences)):
                    word_tokens = tokenizer.encode(
                        target_words[b], add_special_tokens=False
                    )
                    input_ids = inputs["input_ids"][b].tolist()
                    target_indices = _find_sublist(input_ids, word_tokens)
                    if target_indices:
                        vec = hs[b, target_indices].mean(dim=0).cpu().numpy()
                    else:
                        vec = hs[b, 0].cpu().numpy()
                    vecs.append(vec)
                memmaps[layer_name][i : i + len(batch_sentences)] = np.stack(vecs).astype(np.float32)

            if (i // batch_size) % 10 == 0:
                print(f"    {min(i + batch_size, n_words)}/{n_words} words processed")

    for mem in memmaps.values():
        mem.flush()
    _finalize_stream_save(model_dir, model_name, layer_names, n_words, hidden_dim, "sentence")


# --------------------------------------------------------------------------- #
# Word-only extraction (V1 backward compat)
# --------------------------------------------------------------------------- #

def extract_gpt2_word_only(
    words: list[str],
    model_name: str = "gpt2",
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> dict[str, np.ndarray]:
    """V1 mode: extract per-layer hidden states for each word independently."""
    from transformers import AutoTokenizer, GPT2Model

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    model.to(device)

    n_layers = model.config.n_layer
    layer_names = [f"layer_{i}" for i in range(n_layers + 1)]
    all_layer_vecs = {name: [] for name in layer_names}

    with torch.no_grad():
        for i in range(0, len(words), batch_size):
            batch = words[i : i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            attention_mask = inputs["attention_mask"]
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

            mask = attention_mask.unsqueeze(-1).float()
            for layer_idx, layer_name in enumerate(layer_names):
                hs = hidden_states[layer_idx]
                vecs = (hs * mask).sum(dim=1) / mask.sum(dim=1)
                all_layer_vecs[layer_name].append(vecs.cpu().numpy())

            if (i // batch_size) % 5 == 0:
                print(f"    {min(i + batch_size, len(words))}/{len(words)} words processed")

    return {name: np.concatenate(vecs, axis=0) for name, vecs in all_layer_vecs.items()}


def extract_bert_word_only(
    words: list[str],
    model_name: str = "bert-base-uncased",
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> dict[str, np.ndarray]:
    """V1 mode: extract per-layer hidden states for BERT independently."""
    from transformers import AutoModel, AutoTokenizer

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    model.to(device)

    n_layers = getattr(model.config, "num_hidden_layers", 12)
    layer_names = [f"layer_{i}" for i in range(n_layers + 1)]
    all_layer_vecs = {name: [] for name in layer_names}

    with torch.no_grad():
        for i in range(0, len(words), batch_size):
            batch = words[i : i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            attention_mask = inputs["attention_mask"]
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

            mask = attention_mask.unsqueeze(-1).float()
            for layer_idx, layer_name in enumerate(layer_names):
                hs = hidden_states[layer_idx]
                vecs = (hs * mask).sum(dim=1) / mask.sum(dim=1)
                all_layer_vecs[layer_name].append(vecs.cpu().numpy())

            if (i // batch_size) % 5 == 0:
                print(f"    {min(i + batch_size, len(words))}/{len(words)} words processed")

    return {name: np.concatenate(vecs, axis=0) for name, vecs in all_layer_vecs.items()}


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #

MODEL_FAMILIES = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2",
    "bert-base-uncased": "bert",
    # German models for MEG-SCANS
    "dbmdz/german-gpt2": "gpt2",
    "bert-base-german-cased": "bert",
    # Multilingual baseline
    "bert-base-multilingual-cased": "bert",
}

CONTEXTUAL_EXTRACTORS = {
    "gpt2": extract_gpt2_contextual,
    "bert": extract_bert_contextual,
}

CONTEXTUAL_STREAM_EXTRACTORS = {
    "gpt2": extract_gpt2_contextual_to_disk,
    "bert": extract_bert_contextual_to_disk,
}

WORD_ONLY_EXTRACTORS = {
    "gpt2": extract_gpt2_word_only,
    "bert": extract_bert_word_only,
}


def save_embeddings(
    layer_vecs: dict[str, np.ndarray],
    output_dir: Path,
    model_name: str,
    words: list[str],
    context_mode: str = "word",
) -> None:
    model_dir = output_dir / model_name.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    layer_info = []
    for layer_name, vecs in layer_vecs.items():
        np.save(model_dir / f"{layer_name}.npy", vecs.astype(np.float32))
        layer_info.append({"name": layer_name, "shape": list(vecs.shape)})

    metadata = {
        "model_name": model_name,
        "n_words": len(words),
        "context_mode": context_mode,
        "layers": layer_info,
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved {len(layer_info)} layers to {model_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract context-aware LLM layer embeddings (OscilAlign V2)"
    )
    parser.add_argument("--stimuli_file", type=str, required=True,
                        help="Path to word list file (one word per line)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for embeddings")
    parser.add_argument("--events_dir", type=str, default=None,
                        help="Root data dir with events TSV (for sentence context mode)")
    parser.add_argument("--trial_metadata_file", type=str, default=None,
                        help="Optional trial metadata TSV from preprocessing "
                             "(preferred for exact session/task matching)")
    parser.add_argument("--models", nargs="+",
                        default=["gpt2", "gpt2-medium", "bert-base-uncased"],
                        help="Model names to extract from")
    parser.add_argument("--context_mode", type=str, default="sentence",
                        choices=["word", "sentence"],
                        help="'sentence' = context-aware (V2), 'word' = word-only (V1)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--min_word_len", type=int, default=2,
                        help="Min word length (must match preprocessing)")
    parser.add_argument("--stream_save", action="store_true",
                        help="Write layer files incrementally to disk "
                             "(recommended for very large datasets)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    words = load_words(args.stimuli_file)
    print(f"Loaded {len(words)} words from {args.stimuli_file}")
    print(f"  First 10 words: {words[:10]}")

    # Build sentence contexts if needed
    contexts = None
    if args.context_mode == "sentence":
        trial_meta_path = None
        if args.trial_metadata_file is not None:
            trial_meta_path = Path(args.trial_metadata_file)
        else:
            inferred = Path(args.stimuli_file).with_name("trial_metadata.tsv")
            if inferred.exists():
                trial_meta_path = inferred

        if trial_meta_path is not None and trial_meta_path.exists():
            print(f"\nReconstructing sentence contexts from {trial_meta_path}...")
            contexts = build_sentence_contexts_from_trial_metadata(trial_meta_path)

        if not contexts:
            if args.events_dir is None:
                print(
                    "WARNING: sentence mode requested but no usable trial metadata "
                    "or --events_dir was provided; falling back to word mode"
                )
                args.context_mode = "word"
            else:
                print(f"\nReconstructing sentence contexts from {args.events_dir}...")
                contexts = build_sentence_contexts(
                    words, Path(args.events_dir), min_word_len=args.min_word_len
                )

        if contexts:
            print(f"  Built contexts for {len(contexts)} words")
            print(f"  Example: word='{contexts[0]['word']}', "
                  f"sentence='{contexts[0]['sentence'][:80]}...'")

    device = get_device()
    print(f"Using device: {device}")

    for model_name in args.models:
        print(f"\nExtracting embeddings from {model_name} "
              f"(context_mode={args.context_mode})...")

        family = MODEL_FAMILIES.get(model_name)
        if family is None:
            print(f"  WARNING: unsupported model {model_name}, skipping")
            continue

        if args.context_mode == "sentence" and contexts is not None:
            if args.stream_save:
                extractor = CONTEXTUAL_STREAM_EXTRACTORS[family]
                extractor(
                    contexts,
                    output_dir,
                    model_name=model_name,
                    batch_size=args.batch_size,
                    device=device,
                )
                continue
            extractor = CONTEXTUAL_EXTRACTORS[family]
            layer_vecs = extractor(
                contexts, model_name=model_name,
                batch_size=args.batch_size, device=device,
            )
            result_words = [ctx["word"] for ctx in contexts]
        else:
            if args.stream_save:
                print("  WARNING: --stream_save is only implemented for sentence mode; "
                      "falling back to in-memory saving")
            extractor = WORD_ONLY_EXTRACTORS[family]
            layer_vecs = extractor(
                words, model_name=model_name,
                batch_size=args.batch_size, device=device,
            )
            result_words = words

        save_embeddings(layer_vecs, output_dir, model_name, result_words,
                        context_mode=args.context_mode)

    print("\nDone. All embeddings saved.")


if __name__ == "__main__":
    main()
