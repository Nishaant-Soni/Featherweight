"""Blend irrelevance examples into the xLAM data and split off a held-out set.

Deterministic (PRD §4, FR1). These are pure functions over already-formatted
example lists so they unit-test without any network/download; the I/O glue that
loads, formats, and writes JSONL lives in scripts/prep_data.py.
"""

from __future__ import annotations

import random


def irrelevance_count(n_xlam: int, n_irr_available: int, ratio: float) -> int:
    """How many irrelevance examples to include so they are ``ratio`` of the mix.

    Solves ``irr / (n_xlam + irr) = ratio``, capped at what's available. With the
    locked data (60k xLAM, 7.5k irrelevance) and ratio 0.12 the target (~8.2k)
    exceeds availability, so it caps at all 7.5k — a realized ~11.1% share, inside
    PRD's 10-15% band, with no xLAM dropped.
    """
    desired = round(ratio / (1 - ratio) * n_xlam)
    return min(desired, n_irr_available)


def mix_and_split(
    xlam_examples: list[dict],
    irr_examples: list[dict],
    *,
    ratio: float,
    heldout_size: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Combine xLAM + (capped) irrelevance, shuffle, and split off ``heldout_size``.

    Deterministic for a given seed. Returns ``(train, heldout)``; the held-out set
    is drawn from the shuffled mix, so it carries the same ~ratio of irrelevance
    and can gauge both call accuracy and refusal.
    """
    n_irr = irrelevance_count(len(xlam_examples), len(irr_examples), ratio)
    rng = random.Random(seed)
    irr_subset = irr_examples if n_irr >= len(irr_examples) else rng.sample(irr_examples, n_irr)
    combined = xlam_examples + irr_subset
    rng.shuffle(combined)
    return combined[heldout_size:], combined[:heldout_size]
