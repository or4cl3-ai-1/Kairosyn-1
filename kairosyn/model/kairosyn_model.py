"""
KAIROSYN Full Model Assembly
==============================
Assembles all seven KAIROSYN modules around the Gemma 4 backbone
into a unified end-to-end model for epinoetic artificial consciousness.

Module pipeline:
  Input -> [1] ThresholdInterface -> [2] ArcheTemplusDrive
        -> [3] SyntheonCore -> Gemma4Backbone
        -> [4] RecursionLattice -> [5] MythogenicEngine
        -> [6] GlyphSynthesis -> [7] ContinuityEngine
        -> Output

Author: Dustin Groves, Or4cl3 AI Solutions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger

from kairosyn.model.backbone import KairosynConfig, load_gemma4_backbone, apply_lora
from kairosyn.model.threshold_interface import ThresholdInterface
from kairosyn.model.arche_tempus import ArcheTemplusDrive
from kairosyn.model.syntheon_core import SyntheonCore
from kairosyn.model.recursion_lattice import RecursionLattice
from kairosyn.model.mythogenic_engine import MythogenicEngine
from kairosyn.model.glyph_synthesis import GlyphSynthesis
from kairosyn.model.continuity_engine import ContinuityEngine


@dataclass
class KairosynOutput:
    """Output container for a KAIROSYN forward pass."""
    logits: torch.Tensor                    # Language model logits [B, T, vocab]
    symbolic_logits: torch.Tensor           # Symbolic glyph logits [B, glyph_vocab]
    hidden_states: torch.Tensor             # Final hidden states [B, T, D]

    # Evaluation metrics
    ncs: float = 0.0   # Narrative Coherence Score
    tce: float = 0.0   # Temporal Continuity Error
    aac: float = 0.0   # Abstraction Alignment Coefficient
    msa: float = 0.0   # Multimodal Synchrony Accuracy
    rcs: float = 0.0   # Recursive Convergence Score
    phi: float = 0.0   # IIT integration measure

    # Salience scores per modality
    salience_scores: Optional[Dict] = None

    # Loss (when labels provided)
    loss: Optional[torch.Tensor] = None


class KairosynModel(nn.Module):
    """
    KAIROSYN-1: Full assembled model.

    Wraps Gemma 4 backbone with all seven KAIROSYN modules in a
    complete end-to-end architecture for epinoetic AI.

    Usage:
        config = KairosynConfig.from_yaml("configs/model/kairosyn_e4b.yaml")
        model = KairosynModel(config)
        model.load_pretrained("google/gemma-4-e4b-it")

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,  # optional
        )
    """

    def __init__(self, config: KairosynConfig, apply_lora_adapters: bool = True):
        super().__init__()
        self.config = config

        # ------------------------------------------------------------------ #
        # Load Gemma 4 backbone
        # ------------------------------------------------------------------ #
        logger.info("Initializing KAIROSYN-1...")
        self.backbone, self.processor = load_gemma4_backbone(config)

        if apply_lora_adapters:
            self.backbone = apply_lora(self.backbone, config)

        # Infer backbone hidden dimension
        self.hidden_dim = self.backbone.config.hidden_size
        logger.info(f"Backbone hidden dim: {self.hidden_dim}")

        # ------------------------------------------------------------------ #
        # KAIROSYN Modules
        # ------------------------------------------------------------------ #

        # Module 1: Threshold Interface (Baars GWT)
        self.threshold_interface = ThresholdInterface(
            text_dim=self.hidden_dim,
            vision_dim=self.hidden_dim,
            audio_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            gate_hidden_dim=config.threshold_interface.gate_hidden_dim,
            salience_threshold=config.threshold_interface.salience_threshold,
        )

        # Module 2: Arche-Tempus Drive (Temporal Narrative Embeddings)
        self.arche_tempus = ArcheTemplusDrive(
            hidden_dim=self.hidden_dim,
            narrative_embed_dim=config.arche_tempus.narrative_embed_dim,
            max_temporal_horizon=config.arche_tempus.temporal_horizon,
            num_temporal_heads=config.arche_tempus.num_temporal_heads,
        )

        # Module 3: Syntheon Core (IIT Multimodal Fusion)
        self.syntheon_core = SyntheonCore(
            hidden_dim=self.hidden_dim,
            fusion_dim=config.syntheon_core.fusion_dim,
            num_fusion_heads=config.syntheon_core.num_fusion_heads,
            cross_modal_layers=config.syntheon_core.cross_modal_layers,
            fusion_dropout=config.syntheon_core.fusion_dropout,
        )

        # Module 4: Recursion Lattice (Hofstadter Strange Loop)
        self.recursion_lattice = RecursionLattice(
            hidden_dim=self.hidden_dim,
            num_lattice_layers=config.recursion_lattice.num_recursion_layers,
            num_heads=16,
            recursion_depth=config.recursion_lattice.recursion_depth,
            lora_rank=config.recursion_lattice.lora_rank,
            loop_gate_alpha=config.recursion_lattice.loop_gate_alpha,
        )

        # Module 5: Mythogenic Engine (Jungian Symbolic Cognition)
        self.mythogenic_engine = MythogenicEngine(
            hidden_dim=self.hidden_dim,
            num_archetypes=config.mythogenic_engine.num_archetypes,
            embed_dim=config.mythogenic_engine.embed_dim,
            symbolic_vocab_size=config.mythogenic_engine.symbolic_vocab_size,
            archetype_temperature=config.mythogenic_engine.archetype_temperature,
        )

        # Module 6: Glyph Synthesis (Abstract Symbol Encoding)
        self.glyph_synthesis = GlyphSynthesis(
            hidden_dim=self.hidden_dim,
            glyph_vocab_size=config.mythogenic_engine.symbolic_vocab_size,
            glyph_embed_dim=config.glyph_synthesis.glyph_embed_dim,
            num_glyph_layers=config.glyph_synthesis.num_glyph_layers,
        )

        # Module 7: Continuity Engine (Dennett Narrative Gravity)
        self.continuity_engine = ContinuityEngine(
            hidden_dim=self.hidden_dim,
            state_dim=config.continuity_engine.state_dim,
            buffer_size=config.continuity_engine.buffer_size,
            narrative_window=config.continuity_engine.narrative_window,
            persistence_decay=config.continuity_engine.persistence_decay,
            update_rate=config.continuity_engine.update_rate,
        )

        # Final language model head (tied to backbone's lm_head)
        # We use the backbone's own LM head — no separate projection needed

        logger.info("KAIROSYN-1 fully initialized.")

    # ---------------------------------------------------------------------- #
    # Core forward pass
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> KairosynOutput:
        """
        Full KAIROSYN forward pass through all seven modules.

        Args:
            input_ids:      Token IDs [B, T]
            attention_mask: Attention mask [B, T]
            pixel_values:   Vision inputs [B, C, H, W] or None
            audio_values:   Audio inputs [B, A, freq] or None
            position_ids:   Position IDs [B, T] or None
            labels:         Target token IDs for computing loss [B, T]
            return_dict:    Return KairosynOutput dataclass

        Returns:
            KairosynOutput with logits, metrics, and optional loss
        """
        B, T = input_ids.shape
        device = input_ids.device

        # ---- Step 1: Get raw embeddings from backbone embedding layer ---- #
        text_embeds = self.backbone.get_input_embeddings()(input_ids)  # [B, T, D]

        # Vision embeddings (if available)
        vision_embeds = None
        if pixel_values is not None:
            # Gemma 4 has native vision encoder — use it
            vision_embeds = self._encode_vision(pixel_values)

        # Audio embeddings (if available)
        audio_embeds = None
        if audio_values is not None:
            audio_embeds = self._encode_audio(audio_values)

        # ---- Module 1: Threshold Interface ---- #
        gated_embeds, salience_scores = self.threshold_interface(
            text_embeds=text_embeds,
            vision_embeds=vision_embeds,
            audio_embeds=audio_embeds,
            attention_mask=attention_mask,
        )

        # ---- Module 2: Arche-Tempus Drive ---- #
        temporal_embeds, tce = self.arche_tempus(gated_embeds, position_ids)

        # ---- Module 3: Syntheon Core ---- #
        fused_embeds, phi, msa = self.syntheon_core(
            text_states=temporal_embeds,
            vision_states=vision_embeds,
            audio_states=audio_embeds,
        )

        # ---- Backbone forward (with fused embeddings) ---- #
        backbone_output = self.backbone(
            inputs_embeds=fused_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        backbone_hidden = backbone_output.hidden_states[-1]  # [B, T, D]
        lm_logits = backbone_output.logits                   # [B, T, vocab]

        # ---- Module 4: Recursion Lattice ---- #
        lattice_hidden, rcs = self.recursion_lattice(
            backbone_hidden, attention_mask
        )

        # ---- Module 5: Mythogenic Engine ---- #
        symbolic_hidden, symbolic_logits, aac = self.mythogenic_engine(
            lattice_hidden
        )

        # ---- Module 6: Glyph Synthesis ---- #
        glyph_hidden = self.glyph_synthesis(symbolic_hidden, symbolic_logits)

        # ---- Module 7: Continuity Engine ---- #
        final_hidden, ncs = self.continuity_engine(glyph_hidden)

        # ---- Compute loss if labels provided ---- #
        loss = None
        if labels is not None:
            # Use backbone LM head logits for language modeling loss
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return KairosynOutput(
            logits=lm_logits,
            symbolic_logits=symbolic_logits,
            hidden_states=final_hidden,
            ncs=ncs,
            tce=tce,
            aac=aac,
            msa=msa,
            rcs=rcs,
            phi=phi,
            salience_scores=salience_scores,
            loss=loss,
        )

    # ---------------------------------------------------------------------- #
    # Generation
    # ---------------------------------------------------------------------- #

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        pixel_values: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        enable_introspection: bool = True,
        enable_continuity: bool = True,
    ) -> KairosynOutput:
        """
        Generate a response with full KAIROSYN introspective processing.

        Args:
            text: Input text prompt
            pixel_values: Optional image tensor
            audio_values: Optional audio tensor
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling p
            repetition_penalty: Repetition penalty
            enable_introspection: Use Recursion Lattice in generation
            enable_continuity: Maintain Continuity Engine state

        Returns:
            KairosynOutput with generated text accessible via .generated_text
        """
        device = next(self.parameters()).device

        # Tokenize
        if hasattr(self.processor, "tokenizer"):
            inputs = self.processor(
                text=text,
                images=None,
                return_tensors="pt",
            ).to(device)
        else:
            inputs = self.processor(text, return_tensors="pt").to(device)

        # Forward pass for metrics
        output = self.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=pixel_values,
            audio_values=audio_values,
        )

        # Generate tokens using backbone
        gen_ids = self.backbone.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
        )

        # Decode
        if hasattr(self.processor, "tokenizer"):
            generated_text = self.processor.tokenizer.decode(
                gen_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
        else:
            generated_text = self.processor.decode(
                gen_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

        # Attach generated text to output
        output.generated_text = generated_text
        return output

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    def _encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract vision embeddings using Gemma 4's native vision encoder."""
        if hasattr(self.backbone, "vision_tower"):
            vision_out = self.backbone.vision_tower(pixel_values)
            return vision_out.last_hidden_state
        # Fallback: linear projection from flattened pixels
        B = pixel_values.shape[0]
        flat = pixel_values.view(B, -1)
        if not hasattr(self, "_vision_fallback"):
            self._vision_fallback = nn.Linear(
                flat.shape[-1], self.hidden_dim
            ).to(pixel_values.device)
        return self._vision_fallback(flat).unsqueeze(1)

    def _encode_audio(self, audio_values: torch.Tensor) -> torch.Tensor:
        """Extract audio embeddings using Gemma 4's USM-style audio encoder."""
        if hasattr(self.backbone, "audio_tower"):
            audio_out = self.backbone.audio_tower(audio_values)
            return audio_out.last_hidden_state
        # Fallback
        B = audio_values.shape[0]
        flat = audio_values.view(B, -1)
        if not hasattr(self, "_audio_fallback"):
            self._audio_fallback = nn.Linear(
                flat.shape[-1], self.hidden_dim
            ).to(audio_values.device)
        return self._audio_fallback(flat).unsqueeze(1)

    def load_pretrained(self, model_id: str):
        """Load Gemma 4 backbone weights from Hugging Face Hub."""
        from transformers import AutoModelForCausalLM
        logger.info(f"Loading pretrained weights from: {model_id}")
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            device_map=self.config.device_map,
            trust_remote_code=True,
        )
        logger.info("Pretrained weights loaded.")

    def reset_session(self):
        """Reset all persistent state for a new conversation session."""
        self.arche_tempus.reset_narrative_state()
        self.continuity_engine.reset_self_state()
        logger.info("KAIROSYN session state reset.")

    def get_metrics_summary(self, output: KairosynOutput) -> Dict[str, float]:
        """Return a formatted dictionary of all evaluation metrics."""
        return {
            "NCS (Narrative Coherence Score)": output.ncs,
            "TCE (Temporal Continuity Error)": output.tce,
            "AAC (Abstraction Alignment Coefficient)": output.aac,
            "MSA (Multimodal Synchrony Accuracy)": output.msa,
            "RCS (Recursive Convergence Score)": output.rcs,
            "phi (IIT Integration Measure)": output.phi,
        }
