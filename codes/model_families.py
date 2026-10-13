"""Single source of truth for grouping the 32 checkpoints into model families.

Why this file exists. Two analyses in this project independently implemented a family() function
and disagreed: one yielded 12 families, the other 14, and BOTH fed the same class-level argument
(the family-clustered permutation on the ENC-RSA rank gap, and the family-clustered bootstrap on
the participation-ratio mechanism). A consistency review caught the mismatch. Since the family is
the unit of independence for every clustered inference we report, it cannot be defined twice.

The disagreement was over exactly two checkpoints, albert-base-v2 and distilbert-base-uncased:
one grouping folded them into "bert", the other split them out.

We fold them in, giving 12 families. Both are trained on the same corpus as BERT and share its
tokenizer -- DistilBERT is literally distilled from BERT -- so treating them as independent
families would overstate how many independent observations we have. Fewer families means fewer
resampling units, wider clustered CIs, and a harder test, which is the right side to err on when
the claims rest on those CIs excluding zero.

    from model_families import family, N_FAMILIES
"""

# order matters: more specific prefixes first, so "roberta"/"deberta"/"distilbert"/"albert"
# are not swallowed by the bare "bert" rule.
_RULES = [
    ("pythia", "pythia"),
    ("gpt-neo", "gpt-neo"),
    ("gpt2", "gpt2"),          # collapses gpt2 / medium / large / xl
    ("opt", "opt"),
    ("qwen", "qwen"),
    ("llama", "llama"),
    ("roberta", "roberta"),
    ("deberta", "deberta"),
    ("electra", "electra"),
    ("t5", "t5"),
    ("bart", "bart"),
    # distilbert and albert fold into bert: same corpus and tokenizer lineage, not independent
    ("distilbert", "bert"),
    ("albert", "bert"),
    ("bert", "bert"),
]

N_FAMILIES = 12  # over the paper's 32-checkpoint set


def family(model: str) -> str:
    """Map a checkpoint name to its family. Unknown names map to themselves."""
    m = model.lower()
    for needle, fam in _RULES:
        if needle in m:
            return fam
    return m
