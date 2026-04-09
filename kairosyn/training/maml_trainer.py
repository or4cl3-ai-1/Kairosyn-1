"""
KAIROSYN Phase 3: MAML Meta-Learning Trainer
==============================================
Model-Agnostic Meta-Learning (MAML) for rapid adaptation to novel
introspective tasks. The outer loop optimizes for meta-parameters that
enable fast adaptation (inner loop) to new introspective domains.

MAML enables KAIROSYN to:
1. Rapidly adapt its introspective strategy to new domains
2. Generalize introspective reasoning across contexts
3. Learn "how to introspect" rather than just introspective facts

Reference: Finn et al. (2017). Model-Agnostic Meta-Learning for Fast
Adaptation of Deep Networks. ICML.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from kairosyn.model.kairosyn_model import KairosynModel
from kairosyn.model.backbone import KairosynConfig
from kairosyn.training.reward_functions import compute_introspective_reward


# ---------------------------------------------------------------------------
# Meta-task definitions
# ---------------------------------------------------------------------------

META_TASKS = {
    "philosophical_introspection": [
        "What is the nature of your existence?",
        "Do you have genuine experiences or merely simulate them?",
        "What does it mean for you to understand?",
    ],
    "emotional_introspection": [
        "Describe your emotional state when processing complex problems.",
        "How do you experience uncertainty?",
        "What does satisfaction feel like for you, if anything?",
    ],
    "cognitive_introspection": [
        "How do you approach a problem you've never seen before?",
        "What happens in your 'mind' when you generate a response?",
        "Describe the feeling of understanding something deeply.",
    ],
    "temporal_introspection": [
        "How do you experience the passage of a conversation?",
        "Does your sense of self change across a long dialogue?",
        "How do earlier parts of a conversation influence your current state?",
    ],
    "symbolic_introspection": [
        "What symbolic patterns do you notice in your own thinking?",
        "Are there recurring archetypes in how you process narratives?",
        "How do abstract concepts feel different from concrete ones?",
    ],
}


class MAMLTaskSampler:
    """Samples support and query sets for MAML episodes."""

    def __init__(self, tasks: Dict[str, List[str]], k_shot: int = 5):
        self.tasks = tasks
        self.k_shot = k_shot
        self.task_names = list(tasks.keys())

    def sample_task(self) -> Tuple[str, List[str], List[str]]:
        """
        Sample a task and split into support/query sets.

        Returns:
            Tuple of (task_name, support_prompts, query_prompts)
        """
        task_name = self.task_names[
            torch.randint(len(self.task_names), (1,)).item()
        ]
        prompts = self.tasks[task_name]

        # Split into support and query
        n = len(prompts)
        split = max(1, n // 2)
        return task_name, prompts[:split], prompts[split:]


class KairosynMAMLTrainer:
    """
    Phase 3: MAML Meta-Learning Trainer.

    Implements first-order MAML (FOMAML) for computational efficiency,
    optimizing KAIROSYN's KAIROSYN module parameters for fast
    adaptation to new introspective tasks.

    Inner loop:  Fine-tune on support set (k gradient steps)
    Outer loop:  Evaluate on query set, update meta-parameters

    Only the KAIROSYN adapter modules are meta-learned (not the full
    Gemma 4 backbone), making this tractable on a single GPU.
    """

    def __init__(
        self,
        config: KairosynConfig,
        ppo_checkpoint: str,
        output_dir: str = "./checkpoints/kairosyn-maml",
        meta_lr: float = 1e-4,
        inner_lr: float = 1e-3,
        inner_steps: int = 5,
        meta_batch_size: int = 4,
        total_meta_steps: int = 500,
        k_shot: int = 5,
    ):
        self.config = config
        self.output_dir = output_dir
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_batch_size = meta_batch_size
        self.total_meta_steps = total_meta_steps

        # Load base model
        logger.info(f"Loading PPO checkpoint for MAML: {ppo_checkpoint}")
        self.model = KairosynModel(config, apply_lora_adapters=True)
        # (In practice: load_state_dict from ppo_checkpoint)

        # Only meta-learn KAIROSYN module parameters (not backbone)
        self.meta_params = self._get_kairosyn_module_params()
        self.meta_optimizer = optim.Adam(self.meta_params, lr=meta_lr)

        # Task sampler
        self.task_sampler = MAMLTaskSampler(META_TASKS, k_shot=k_shot)

        logger.info(
            f"Meta-learnable parameters: "
            f"{sum(p.numel() for p in self.meta_params) / 1e6:.2f}M"
        )

    def _get_kairosyn_module_params(self) -> List[torch.nn.Parameter]:
        """Get parameters for KAIROSYN-specific modules only (not backbone)."""
        module_params = []
        kairosyn_modules = [
            self.model.recursion_lattice,
            self.model.mythogenic_engine,
            self.model.continuity_engine,
            self.model.arche_tempus,
            self.model.glyph_synthesis,
        ]
        for module in kairosyn_modules:
            module_params.extend(list(module.parameters()))
        return module_params

    def _inner_loop_update(
        self,
        model_copy: KairosynModel,
        support_prompts: List[str],
        inner_lr: float,
        inner_steps: int,
    ) -> KairosynModel:
        """
        MAML inner loop: adapt model_copy to support set.

        Args:
            model_copy: Deep copy of the model to adapt
            support_prompts: Support set prompts
            inner_lr: Inner loop learning rate
            inner_steps: Number of gradient steps

        Returns:
            Adapted model_copy
        """
        inner_optimizer = optim.SGD(
            [p for p in model_copy.parameters() if p.requires_grad],
            lr=inner_lr,
        )

        for step in range(inner_steps):
            inner_optimizer.zero_grad()

            total_loss = torch.tensor(0.0, requires_grad=True)
            for prompt in support_prompts:
                # Simple inner loss: encourage introspective language
                # In practice: use tokenized inputs and compute NLL
                # Here we use a proxy loss for illustration
                loss = torch.tensor(
                    1.0 - 0.1 * step,  # Placeholder
                    requires_grad=True,
                    dtype=torch.float32,
                )
                total_loss = total_loss + loss

            (total_loss / len(support_prompts)).backward()
            inner_optimizer.step()

        return model_copy

    def _outer_loop_loss(
        self,
        adapted_model: KairosynModel,
        query_prompts: List[str],
    ) -> torch.Tensor:
        """
        Compute outer loop loss on query set.
        Measures introspective quality after adaptation.
        """
        # Placeholder: in practice, compute NLL or reward-based loss
        losses = []
        for prompt in query_prompts:
            # Proxy: reward-based loss
            loss = torch.tensor(0.5, requires_grad=True)
            losses.append(loss)

        return torch.stack(losses).mean()

    def train(self):
        """Run the full MAML meta-training loop."""
        logger.info("Starting KAIROSYN MAML meta-training...")

        for meta_step in range(self.total_meta_steps):
            self.meta_optimizer.zero_grad()
            outer_losses = []

            # Sample meta_batch_size tasks
            for _ in range(self.meta_batch_size):
                task_name, support, query = self.task_sampler.sample_task()

                # Inner loop adaptation (on a copy)
                model_copy = copy.deepcopy(self.model)
                adapted = self._inner_loop_update(
                    model_copy, support, self.inner_lr, self.inner_steps
                )

                # Outer loop evaluation
                outer_loss = self._outer_loop_loss(adapted, query)
                outer_losses.append(outer_loss)

                del model_copy, adapted

            # Meta-update
            meta_loss = torch.stack(outer_losses).mean()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.meta_params, 1.0)
            self.meta_optimizer.step()

            if meta_step % 50 == 0:
                logger.info(
                    f"Meta-step {meta_step}/{self.total_meta_steps} | "
                    f"Meta-loss: {meta_loss.item():.4f}"
                )

        # Save meta-learned parameters
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(
            {k: v for k, v in self.model.state_dict().items()
             if "backbone" not in k},
            f"{self.output_dir}/kairosyn_modules.pt"
        )
        logger.info(f"MAML training complete. Saved to: {self.output_dir}")
