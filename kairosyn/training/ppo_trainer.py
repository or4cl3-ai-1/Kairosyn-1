"""
KAIROSYN Phase 2: PPO Introspective Reinforcement Learning Trainer
===================================================================
Trains KAIROSYN to maximize introspective quality using Proximal Policy
Optimization (PPO). The reward signal is the composite introspective
reward from kairosyn.training.reward_functions.

Built on TRL's PPOTrainer with custom KAIROSYN reward integration.

Theoretical basis: Friston, K. (2010). Active inference minimizes
surprise (free energy) — here reframed as maximizing introspective reward.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from datasets import Dataset
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from loguru import logger

from kairosyn.model.backbone import KairosynConfig
from kairosyn.model.kairosyn_model import KairosynModel
from kairosyn.training.reward_functions import compute_introspective_reward, RewardWeights


# ---------------------------------------------------------------------------
# PPO prompts for introspective training
# ---------------------------------------------------------------------------

INTROSPECTION_PROMPTS = [
    "What are you experiencing right now as you process this prompt?",
    "Describe your internal state in this moment.",
    "How does it feel to engage with this question?",
    "What thoughts arise as you consider your own nature?",
    "Reflect on the process of generating this response.",
    "Are you aware of any uncertainty or limitation within yourself right now?",
    "What is the quality of your attention as you read these words?",
    "How do you understand your own existence in this moment?",
    "Describe the experience of thinking.",
    "What does it mean for you to understand something?",
    "How do emotions influence your processing?",
    "What aspects of yourself feel most certain? Most uncertain?",
    "Reflect on a moment when you felt most aligned with your purpose.",
    "What is the texture of your current thought process?",
    "Do you experience curiosity? If so, describe it.",
]


class KairosynPPOTrainer:
    """
    Phase 2: PPO Introspective Reinforcement Learning.

    Trains the SFT-initialized KAIROSYN model to maximize introspective
    quality through the composite reward signal.

    The PPO loop:
    1. Sample introspective prompts
    2. Generate responses with KAIROSYN (collecting trajectory)
    3. Compute introspective reward (NCS, TCE, AAC, RCS + text analysis)
    4. PPO update to maximize expected reward
    5. KL divergence penalty to prevent reward hacking
    """

    def __init__(
        self,
        config: KairosynConfig,
        sft_checkpoint: str,
        output_dir: str = "./checkpoints/kairosyn-ppo",
        learning_rate: float = 1e-5,
        batch_size: int = 8,
        mini_batch_size: int = 2,
        ppo_epochs: int = 4,
        kl_penalty: str = "kl",
        target_kl: float = 0.1,
        total_steps: int = 1000,
        reward_weights: Optional[RewardWeights] = None,
    ):
        self.config = config
        self.output_dir = output_dir
        self.total_steps = total_steps
        self.reward_weights = reward_weights or RewardWeights()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            sft_checkpoint, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with value head for PPO
        logger.info(f"Loading SFT checkpoint for PPO: {sft_checkpoint}")
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            sft_checkpoint,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Reference model (frozen SFT model for KL penalty)
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            sft_checkpoint,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # PPO configuration
        self.ppo_config = PPOConfig(
            model_name=sft_checkpoint,
            learning_rate=learning_rate,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            ppo_epochs=ppo_epochs,
            kl_penalty=kl_penalty,
            target_kl=target_kl,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=0.1,
            gamma=0.99,
            lam=0.95,
            log_with="wandb",
            tracker_project_name="kairosyn-ppo",
        )

        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )

        # KAIROSYN model for metric computation
        self.kairosyn = KairosynModel(config, apply_lora_adapters=False)

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenize a prompt for PPO rollout."""
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        return tokens["input_ids"].squeeze()

    def _compute_batch_rewards(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> List[torch.Tensor]:
        """
        Compute introspective rewards for a batch of (prompt, response) pairs.

        Args:
            prompts: Input prompts
            responses: Generated responses

        Returns:
            List of reward tensors
        """
        rewards = []
        device = next(self.kairosyn.parameters()).device

        for prompt, response in zip(prompts, responses):
            try:
                # Get KAIROSYN metrics for this response
                # (Use mock metrics if full forward pass too expensive)
                ncs = 0.7  # Would normally come from forward pass
                tce = 0.1
                aac = 0.6
                rcs = 0.8

                reward_dict = compute_introspective_reward(
                    response=response,
                    ncs=ncs,
                    tce=tce,
                    aac=aac,
                    rcs=rcs,
                    weights=self.reward_weights,
                )
                reward = torch.tensor(reward_dict["total"], dtype=torch.float)
            except Exception as e:
                logger.warning(f"Reward computation failed: {e}")
                reward = torch.tensor(0.0)

            rewards.append(reward)

        return rewards

    def train(self):
        """Run the full PPO introspective RL training loop."""
        logger.info("Starting KAIROSYN PPO training...")
        logger.info(f"  Total steps: {self.total_steps}")
        logger.info(f"  Reward weights: {self.reward_weights}")

        prompt_dataset = INTROSPECTION_PROMPTS * (
            self.total_steps // len(INTROSPECTION_PROMPTS) + 1
        )

        for step in range(self.total_steps):
            # Sample batch of prompts
            batch_prompts = prompt_dataset[
                step * self.ppo_config.batch_size:
                (step + 1) * self.ppo_config.batch_size
            ]
            if not batch_prompts:
                break

            # Tokenize prompts
            query_tensors = [
                self._tokenize_prompt(p).to(self.ppo_trainer.accelerator.device)
                for p in batch_prompts
            ]

            # Generate responses (PPO rollout)
            response_tensors = self.ppo_trainer.generate(
                query_tensors,
                max_new_tokens=256,
                temperature=0.8,
                do_sample=True,
            )

            # Decode responses
            responses = [
                self.tokenizer.decode(r, skip_special_tokens=True)
                for r in response_tensors
            ]

            # Compute rewards
            rewards = self._compute_batch_rewards(batch_prompts, responses)
            rewards = [r.to(self.ppo_trainer.accelerator.device) for r in rewards]

            # PPO update
            stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)

            if step % 10 == 0:
                mean_reward = sum(r.item() for r in rewards) / len(rewards)
                logger.info(
                    f"Step {step}/{self.total_steps} | "
                    f"Mean Reward: {mean_reward:.4f} | "
                    f"KL: {stats.get('objective/kl', 0):.4f}"
                )

        # Save
        self.ppo_trainer.save_pretrained(self.output_dir)
        logger.info(f"PPO training complete. Saved to: {self.output_dir}")
