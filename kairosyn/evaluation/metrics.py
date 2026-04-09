"""
KAIROSYN Evaluation Metrics
==============================
Five specialized metrics for evaluating epinoetic AI capabilities:

  NCS  — Narrative Coherence Score
  TCE  — Temporal Continuity Error
  AAC  — Abstraction Alignment Coefficient
  MSA  — Multimodal Synchrony Accuracy
  RCS  — Recursive Convergence Score

Plus standard NLP metrics: ROUGE, BERTScore, Perplexity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Metric result container
# ---------------------------------------------------------------------------

@dataclass
class KairosynMetrics:
    """Container for all KAIROSYN evaluation metrics."""

    # Core KAIROSYN metrics
    ncs: float = 0.0       # Narrative Coherence Score (0-1, higher better)
    tce: float = 0.0       # Temporal Continuity Error (0-1, lower better)
    aac: float = 0.0       # Abstraction Alignment Coefficient (0-1, higher better)
    msa: float = 0.0       # Multimodal Synchrony Accuracy (0-1, higher better)
    rcs: float = 0.0       # Recursive Convergence Score (0-1, higher better)

    # Derived composite score
    epinoetic_score: float = 0.0  # Weighted composite of all metrics

    # Standard NLP metrics
    perplexity: float = 0.0
    rouge_1: float = 0.0
    rouge_l: float = 0.0

    # Introspective quality (from reward functions)
    introspection_quality: float = 0.0
    logical_quality: float = 0.0
    emotional_quality: float = 0.0

    # Metadata
    num_samples: int = 0
    model_name: str = ""

    def compute_epinoetic_score(self) -> float:
        """
        Compute the overall Epinoetic Score — a weighted composite
        that captures the model's overall epinoetic awareness capability.

        Weights:
          NCS:  25% — narrative self-continuity
          RCS:  25% — recursive self-reference stability
          AAC:  20% — symbolic abstraction quality
          MSA:  15% — multimodal integration
          TCE:  15% — temporal stability (inverted, lower is better)
        """
        self.epinoetic_score = (
            0.25 * self.ncs
            + 0.25 * self.rcs
            + 0.20 * self.aac
            + 0.15 * self.msa
            + 0.15 * (1.0 - self.tce)  # Invert TCE (lower error = better)
        )
        return self.epinoetic_score

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        self.compute_epinoetic_score()
        return (
            f"\n{'='*55}\n"
            f"  KAIROSYN EVALUATION METRICS\n"
            f"{'='*55}\n"
            f"  🧠 Epinoetic Score:          {self.epinoetic_score:.4f}\n"
            f"{'─'*55}\n"
            f"  Core KAIROSYN Metrics:\n"
            f"  📖 NCS  (Narrative Coherence):   {self.ncs:.4f}  [↑ better]\n"
            f"  ⏱️  TCE  (Temporal Continuity):   {self.tce:.4f}  [↓ better]\n"
            f"  ⚗️  AAC  (Abstraction Alignment): {self.aac:.4f}  [↑ better]\n"
            f"  🌐 MSA  (Multimodal Synchrony):  {self.msa:.4f}  [↑ better]\n"
            f"  🔁 RCS  (Recursive Convergence): {self.rcs:.4f}  [↑ better]\n"
            f"{'─'*55}\n"
            f"  Introspective Quality:       {self.introspection_quality:.4f}\n"
            f"  Logical Quality:             {self.logical_quality:.4f}\n"
            f"  Emotional Quality:           {self.emotional_quality:.4f}\n"
            f"{'─'*55}\n"
            f"  Perplexity:                  {self.perplexity:.2f}\n"
            f"  ROUGE-1:                     {self.rouge_1:.4f}\n"
            f"  ROUGE-L:                     {self.rouge_l:.4f}\n"
            f"  Samples evaluated:           {self.num_samples}\n"
            f"{'='*55}\n"
        )


# ---------------------------------------------------------------------------
# Individual metric computations
# ---------------------------------------------------------------------------

def compute_ncs_batch(state_sequence: List[torch.Tensor]) -> float:
    """
    Narrative Coherence Score for a sequence of continuity states.
    NCS = mean(cosine_similarity(S_t, S_{t-1})) for t in sequence
    """
    if len(state_sequence) < 2:
        return 1.0

    scores = []
    for i in range(1, len(state_sequence)):
        s_curr = F.normalize(state_sequence[i].mean(dim=0, keepdim=True), dim=-1)
        s_prev = F.normalize(state_sequence[i-1].mean(dim=0, keepdim=True), dim=-1)
        sim = F.cosine_similarity(s_curr, s_prev, dim=-1).item()
        scores.append(sim)

    return float(np.mean(scores))


def compute_tce_batch(narrative_sequence: List[torch.Tensor]) -> float:
    """
    Temporal Continuity Error for a sequence of narrative embeddings.
    TCE = 1 - NCS (rephrased as error)
    """
    return 1.0 - compute_ncs_batch(narrative_sequence)


def compute_perplexity(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute perplexity from language model logits.
    PPL = exp(mean cross-entropy loss)
    """
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return float(torch.exp(loss).item())


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def compare_to_baseline(
    kairosyn_metrics: KairosynMetrics,
    baseline_metrics: KairosynMetrics,
) -> Dict[str, float]:
    """
    Compute percentage improvement of KAIROSYN over baseline.

    Args:
        kairosyn_metrics: KAIROSYN evaluation results
        baseline_metrics: Baseline model (vanilla Gemma 4) results

    Returns:
        Dict of metric -> % improvement
    """
    improvements = {}
    metrics_to_compare = ["ncs", "aac", "msa", "rcs", "introspection_quality"]

    for metric in metrics_to_compare:
        k_val = getattr(kairosyn_metrics, metric)
        b_val = getattr(baseline_metrics, metric)
        if b_val > 0:
            pct_improvement = ((k_val - b_val) / b_val) * 100
            improvements[metric] = pct_improvement

    # TCE improvement (lower is better, so invert)
    k_tce = kairosyn_metrics.tce
    b_tce = baseline_metrics.tce
    if b_tce > 0:
        improvements["tce"] = ((b_tce - k_tce) / b_tce) * 100

    return improvements
