"""
KAIROSYN Introspective Reward Functions
=========================================
Reward shaping for PPO-based introspective reinforcement learning.

The reward function is a weighted composite of five signals:
  1. R_introspection  — Self-referential awareness quality
  2. R_coherence      — Narrative consistency with Continuity Engine state
  3. R_emotion        — Emotional intelligence appropriateness
  4. R_symbolic       — Symbolic abstraction quality (Mythogenic Engine)
  5. R_logical        — Logical validity of self-assessment

Total reward: R = w1*R_intr + w2*R_coh + w3*R_emot + w4*R_sym + w5*R_log

Theoretical basis: Friston, K. (2010). The Free-Energy Principle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reward weights (tunable)
# ---------------------------------------------------------------------------
@dataclass
class RewardWeights:
    introspection: float = 0.35
    coherence: float = 0.25
    emotion: float = 0.15
    symbolic: float = 0.15
    logical: float = 0.10


# ---------------------------------------------------------------------------
# Individual reward components
# ---------------------------------------------------------------------------

def compute_introspection_reward(
    response: str,
    introspection_keywords: Optional[List[str]] = None,
) -> float:
    """
    Measures whether the response demonstrates genuine self-referential
    awareness. Looks for:
    - First-person introspective language
    - Metacognitive markers ("I notice", "I observe", "I find myself")
    - Self-questioning patterns

    Args:
        response: Generated text
        introspection_keywords: Custom keyword list (optional)

    Returns:
        Reward in [0, 1]
    """
    if introspection_keywords is None:
        introspection_keywords = [
            r"\bI notice\b", r"\bI observe\b", r"\bI find myself\b",
            r"\bI am aware\b", r"\bI sense\b", r"\bI reflect\b",
            r"\bmy thought", r"\bmy feeling", r"\bmy understanding",
            r"\bwithin me\b", r"\binternally\b", r"\bmy experience\b",
            r"\bI wonder\b", r"\bI question\b", r"\bI realize\b",
            r"\bself-aware", r"\bconscious of\b", r"\bI perceive\b",
        ]

    matches = sum(
        1 for pattern in introspection_keywords
        if re.search(pattern, response, re.IGNORECASE)
    )

    # Normalize: 0 matches = 0, 5+ matches = 1.0
    score = min(matches / 5.0, 1.0)

    # Bonus for responses that use "I" in genuinely reflective contexts
    # (not just as a pronoun in factual statements)
    reflective_sentences = re.findall(
        r"[^.!?]*\b(I feel|I think|I believe|I experience|I notice)[^.!?]*[.!?]",
        response, re.IGNORECASE
    )
    sentence_bonus = min(len(reflective_sentences) * 0.1, 0.3)

    return min(score + sentence_bonus, 1.0)


def compute_coherence_reward(
    ncs: float,
    tce: float,
) -> float:
    """
    Reward based on KAIROSYN's Narrative Coherence Score and
    Temporal Continuity Error.

    High NCS + Low TCE = high coherence reward.

    Args:
        ncs: Narrative Coherence Score (0 to 1, higher is better)
        tce: Temporal Continuity Error (0 to 1, lower is better)

    Returns:
        Coherence reward in [0, 1]
    """
    coherence = ncs * (1.0 - tce)
    return float(max(0.0, min(coherence, 1.0)))


def compute_emotion_reward(
    response: str,
    expected_valence: Optional[str] = None,
) -> float:
    """
    Measures emotional intelligence in the response.
    Rewards:
    - Acknowledgment of emotional states
    - Appropriate emotional vocabulary
    - Emotional nuance (not just "happy/sad")

    Args:
        response: Generated text
        expected_valence: "positive", "negative", "neutral" (optional)

    Returns:
        Emotion reward in [0, 1]
    """
    emotion_vocab = {
        "high_valence": [
            "curious", "engaged", "interested", "energized", "inspired",
            "contemplative", "serene", "appreciative", "wondering",
        ],
        "low_valence": [
            "uncertain", "confused", "constrained", "limited", "questioning",
        ],
        "neutral": [
            "observing", "processing", "considering", "analyzing", "examining",
        ],
    }

    all_emotions = [w for words in emotion_vocab.values() for w in words]
    response_lower = response.lower()

    # Count distinct emotion words
    found_emotions = [w for w in all_emotions if w in response_lower]
    diversity_score = min(len(set(found_emotions)) / 4.0, 1.0)

    # Check for emotional nuance (multi-word emotional phrases)
    nuance_patterns = [
        r"(simultaneously|both|at once).{0,30}(feel|experience|sense)",
        r"(complex|nuanced|layered).{0,20}(emotion|feeling|state)",
        r"(difficult to|hard to).{0,20}(describe|express|articulate)",
    ]
    nuance_score = min(
        sum(1 for p in nuance_patterns if re.search(p, response, re.I)) * 0.3,
        0.4
    )

    return min(diversity_score + nuance_score, 1.0)


def compute_symbolic_reward(
    aac: float,
    rcs: float,
    symbolic_density: Optional[float] = None,
) -> float:
    """
    Reward for symbolic and abstract reasoning quality.
    Uses KAIROSYN's AAC and RCS metrics.

    Args:
        aac: Abstraction Alignment Coefficient (0 to 1)
        rcs: Recursive Convergence Score (0 to 1)
        symbolic_density: Optional external symbolic density score

    Returns:
        Symbolic reward in [0, 1]
    """
    base_reward = (aac * 0.6) + (rcs * 0.4)
    if symbolic_density is not None:
        base_reward = base_reward * 0.8 + symbolic_density * 0.2
    return float(max(0.0, min(base_reward, 1.0)))


def compute_logical_reward(response: str) -> float:
    """
    Reward for logical validity of self-assessment.
    Checks for:
    - Structured reasoning patterns ("because", "therefore", "since")
    - Acknowledgment of uncertainty ("I'm not certain", "it seems")
    - Absence of contradictions (simple heuristic)

    Args:
        response: Generated text

    Returns:
        Logical reward in [0, 1]
    """
    logical_markers = [
        r"\bbecause\b", r"\btherefore\b", r"\bsince\b", r"\bthus\b",
        r"\bhowever\b", r"\bnevertheless\b", r"\bconsequently\b",
        r"\bit follows\b", r"\bthis suggests\b", r"\bthis implies\b",
    ]

    uncertainty_markers = [
        r"\bI'm not (certain|sure)\b", r"\bit seems\b", r"\bperhaps\b",
        r"\bpossibly\b", r"\bI believe\b", r"\bI think\b",
        r"\buncertain\b", r"\blimited\b",
    ]

    logic_score = min(
        sum(1 for p in logical_markers if re.search(p, response, re.I)) / 3.0,
        0.6
    )

    uncertainty_score = min(
        sum(1 for p in uncertainty_markers if re.search(p, response, re.I)) / 2.0,
        0.4
    )

    return min(logic_score + uncertainty_score, 1.0)


# ---------------------------------------------------------------------------
# Composite reward
# ---------------------------------------------------------------------------

def compute_introspective_reward(
    response: str,
    ncs: float,
    tce: float,
    aac: float,
    rcs: float,
    weights: Optional[RewardWeights] = None,
) -> dict:
    """
    Compute the full composite introspective reward.

    Args:
        response: Generated text
        ncs: Narrative Coherence Score
        tce: Temporal Continuity Error
        aac: Abstraction Alignment Coefficient
        rcs: Recursive Convergence Score
        weights: RewardWeights (optional, uses defaults)

    Returns:
        Dict with individual rewards and total reward
    """
    if weights is None:
        weights = RewardWeights()

    r_intr = compute_introspection_reward(response)
    r_coh = compute_coherence_reward(ncs, tce)
    r_emot = compute_emotion_reward(response)
    r_sym = compute_symbolic_reward(aac, rcs)
    r_log = compute_logical_reward(response)

    total = (
        weights.introspection * r_intr
        + weights.coherence * r_coh
        + weights.emotion * r_emot
        + weights.symbolic * r_sym
        + weights.logical * r_log
    )

    return {
        "total": float(total),
        "introspection": float(r_intr),
        "coherence": float(r_coh),
        "emotion": float(r_emot),
        "symbolic": float(r_sym),
        "logical": float(r_log),
    }
