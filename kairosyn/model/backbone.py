"""
KAIROSYN Backbone: Gemma 4 Loader and Configuration
=====================================================
Handles loading, configuration, and preparation of the Gemma 4
foundation model for KAIROSYN module integration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal

import torch
import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoProcessor,
)
from peft import LoraConfig, get_peft_model, TaskType
from loguru import logger


GEMMA4_VARIANTS = {
    "e2b": "google/gemma-4-e2b-it",
    "e4b": "google/gemma-4-e4b-it",
    "26b_moe": "google/gemma-4-26b-moe-it",
    "31b": "google/gemma-4-31b-it",
}


@dataclass
class RecursionLatticeConfig:
    num_recursion_layers: int = 6
    recursion_depth: int = 3
    loop_gate_alpha: float = 0.1
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])


@dataclass
class ContinuityEngineConfig:
    state_dim: int = 2048
    buffer_size: int = 512
    narrative_window: int = 64
    persistence_decay: float = 0.95
    update_rate: float = 0.1


@dataclass
class MythogenicEngineConfig:
    num_archetypes: int = 64
    symbolic_vocab_size: int = 4096
    archetype_temperature: float = 0.8
    embed_dim: int = 1024


@dataclass
class ArcheTemplusConfig:
    temporal_rope_base: int = 1_000_000
    narrative_embed_dim: int = 512
    temporal_horizon: int = 256_000
    num_temporal_heads: int = 8


@dataclass
class ThresholdInterfaceConfig:
    salience_threshold: float = 0.3
    gate_hidden_dim: int = 512
    modalities: list = field(default_factory=lambda: ["text", "vision", "audio"])


@dataclass
class SyntheonCoreConfig:
    fusion_dim: int = 2048
    num_fusion_heads: int = 16
    fusion_dropout: float = 0.1
    cross_modal_layers: int = 4


@dataclass
class GlyphSynthesisConfig:
    glyph_vocab_size: int = 4096
    glyph_embed_dim: int = 512
    num_glyph_layers: int = 3


@dataclass
class KairosynConfig:
    """
    Master configuration for KAIROSYN-1.
    Controls the Gemma 4 backbone and all seven operational modules.
    """
    backbone_id: str = "google/gemma-4-e4b-it"
    backbone_variant: str = "e4b"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True

    recursion_lattice: RecursionLatticeConfig = field(
        default_factory=RecursionLatticeConfig
    )
    continuity_engine: ContinuityEngineConfig = field(
        default_factory=ContinuityEngineConfig
    )
    mythogenic_engine: MythogenicEngineConfig = field(
        default_factory=MythogenicEngineConfig
    )
    arche_tempus: ArcheTemplusConfig = field(
        default_factory=ArcheTemplusConfig
    )
    threshold_interface: ThresholdInterfaceConfig = field(
        default_factory=ThresholdInterfaceConfig
    )
    syntheon_core: SyntheonCoreConfig = field(
        default_factory=SyntheonCoreConfig
    )
    glyph_synthesis: GlyphSynthesisConfig = field(
        default_factory=GlyphSynthesisConfig
    )

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    @classmethod
    def from_yaml(cls, path: str) -> "KairosynConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "KairosynConfig":
        cfg = cls()
        model_cfg = d.get("model", {})
        cfg.backbone_id = model_cfg.get("backbone", cfg.backbone_id)
        cfg.backbone_variant = model_cfg.get("backbone_variant", cfg.backbone_variant)
        if "recursion_lattice" in d:
            cfg.recursion_lattice = RecursionLatticeConfig(**d["recursion_lattice"])
        if "continuity_engine" in d:
            cfg.continuity_engine = ContinuityEngineConfig(**d["continuity_engine"])
        if "mythogenic_engine" in d:
            cfg.mythogenic_engine = MythogenicEngineConfig(**d["mythogenic_engine"])
        if "arche_tempus" in d:
            cfg.arche_tempus = ArcheTemplusConfig(**d["arche_tempus"])
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backbone_id": self.backbone_id,
            "backbone_variant": self.backbone_variant,
            "recursion_lattice": self.recursion_lattice.__dict__,
            "continuity_engine": self.continuity_engine.__dict__,
            "mythogenic_engine": self.mythogenic_engine.__dict__,
            "arche_tempus": self.arche_tempus.__dict__,
        }


def get_bnb_config(cfg: KairosynConfig) -> Optional[BitsAndBytesConfig]:
    if not cfg.use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.use_nested_quant,
    )


def load_gemma4_backbone(
    cfg: KairosynConfig,
    load_in_4bit: Optional[bool] = None,
) -> tuple:
    """
    Load Gemma 4 backbone model and tokenizer/processor.
    Returns tuple of (model, processor).
    """
    use_4bit = load_in_4bit if load_in_4bit is not None else cfg.use_4bit
    bnb_config = get_bnb_config(cfg) if use_4bit else None
    dtype = getattr(torch, cfg.torch_dtype)

    logger.info(f"Loading Gemma 4 backbone: {cfg.backbone_id}")

    try:
        processor = AutoProcessor.from_pretrained(
            cfg.backbone_id, trust_remote_code=True
        )
    except Exception:
        processor = AutoTokenizer.from_pretrained(
            cfg.backbone_id, trust_remote_code=True
        )

    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": cfg.device_map,
        "trust_remote_code": True,
        "attn_implementation": "flash_attention_2",
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(cfg.backbone_id, **model_kwargs)
    model.config.use_cache = False

    logger.info(
        f"Backbone loaded: "
        f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters"
    )
    return model, processor


def apply_lora(model: torch.nn.Module, cfg: KairosynConfig) -> torch.nn.Module:
    """Apply LoRA adapters to Gemma 4 for KAIROSYN fine-tuning."""
    rl_cfg = cfg.recursion_lattice
    lora_config = LoraConfig(
        r=rl_cfg.lora_rank,
        lora_alpha=rl_cfg.lora_alpha,
        lora_dropout=rl_cfg.lora_dropout,
        target_modules=rl_cfg.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
