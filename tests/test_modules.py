"""
KAIROSYN-1 Unit Tests — All Seven Modules
==========================================
CPU-only tests that verify each module's shape contracts,
metric outputs, and basic forward-pass correctness.

Run:
    pytest tests/test_modules.py -v --timeout=120
"""

import pytest
import torch
import torch.nn.functional as F

# ── Fixtures ──────────────────────────────────────────────────────────────────

BATCH  = 2
SEQ    = 16
DIM    = 64    # Small hidden dim for fast CPU tests
HEADS  = 4


@pytest.fixture(scope="module")
def dummy_input():
    return torch.randn(BATCH, SEQ, DIM)


@pytest.fixture(scope="module")
def dummy_single():
    return torch.randn(BATCH, DIM)


# ── Module 1: ThresholdInterface ─────────────────────────────────────────────

class TestThresholdInterface:

    @pytest.fixture(autouse=True)
    def setup(self):
        from kairosyn.model.threshold_interface import ThresholdInterface
        self.module = ThresholdInterface(
            text_dim=DIM, vision_dim=DIM, audio_dim=DIM,
            hidden_dim=DIM, gate_hidden_dim=32, salience_threshold=0.3,
            num_cross_modal_heads=HEADS,
        )

    def test_text_only_output_shape(self, dummy_input):
        out, salience = self.module(dummy_input)
        assert out.shape == (BATCH, SEQ, DIM), \
            f"Expected ({BATCH},{SEQ},{DIM}), got {out.shape}"

    def test_salience_keys(self, dummy_input):
        _, salience = self.module(dummy_input)
        assert "text" in salience

    def test_multimodal_output_shape(self, dummy_input):
        vision = torch.randn(BATCH, 4, DIM)
        audio  = torch.randn(BATCH, 8, DIM)
        out, salience = self.module(dummy_input, vision, audio)
        assert out.shape == (BATCH, SEQ, DIM)
        assert "vision" in salience
        assert "audio"  in salience

    def test_output_not_nan(self, dummy_input):
        out, _ = self.module(dummy_input)
        assert not torch.isnan(out).any(), "Output contains NaN"

    def test_salience_range(self, dummy_input):
        _, salience = self.module(dummy_input)
        scores = salience["text"]
        assert scores.min() >= 0.0 and scores.max() <= 1.0, \
            "Salience scores must be in [0, 1]"


# ── Module 2: ArcheTemplusDrive ──────────────────────────────────────────────

class TestArcheTemplusDrive:

    @pytest.fixture(autouse=True)
    def setup(self):
        from kairosyn.model.arche_tempus import ArcheTemplusDrive
        self.module = ArcheTemplusDrive(
            hidden_dim=DIM, narrative_embed_dim=32,
            max_temporal_horizon=1024, num_temporal_heads=4,
        )

    def test_output_shape(self, dummy_input):
        out, tce = self.module(dummy_input)
        assert out.shape == dummy_input.shape

    def test_tce_is_float(self, dummy_input):
        _, tce = self.module(dummy_input)
        assert isinstance(tce, float)

    def test_tce_first_pass_is_zero(self, dummy_input):
        self.module.reset_narrative_state()
        _, tce = self.module(dummy_input)
        assert tce == 0.0, "First pass TCE should be 0 (no prior state)"

    def test_tce_second_pass_in_range(self, dummy_input):
        self.module.reset_narrative_state()
        self.module(dummy_input)
        _, tce = self.module(torch.randn(BATCH, SEQ, DIM))
        assert 0.0 <= tce <= 2.0, f"TCE out of expected range: {tce}"

    def test_output_not_nan(self, dummy_input):
        out, _ = self.module(dummy_input)
        assert not torch.isnan(out).any()


# ── Module 3: SyntheonCore ───────────────────────────────────────────────────

class TestSyntheonCore:

    @pytest.fixture(autouse=True)
    def setup(self):
        from kairosyn.model.syntheon_core import SyntheonCore
        self.module = SyntheonCore(
            hidden_dim=DIM, fusion_dim=DIM, num_fusion_heads=HEADS,
            cross_modal_layers=2, fusion_dropout=0.0,
        )

    def test_text_only_output_shape(self, dummy_input):
        out, phi, msa = self.module(dummy_input)
        assert out.shape == dummy_input.shape

    def test_phi_is_float(self, dummy_input):
        _, phi, _ = self.module(dummy_input)
        assert isinstance(phi, float)

    def test_msa_is_float(self, dummy_input):
        _, _, msa = self.module(dummy_input)
        assert isinstance(msa, float)

    def test_msa_range(self, dummy_input):
        _, _, msa = self.module(dummy_input)
        assert -1.0 <= msa <= 1.0, f"MSA out of range: {msa}"

    def test_multimodal_output_shape(self, dummy_input):
        vision = torch.randn(BATCH, 4, DIM)
        audio  = torch.randn(BATCH, 6, DIM)
        out, phi, msa = self.module(dummy_input, vision, audio)
        assert out.shape == dummy_input.shape


# ── Module 4: RecursionLattice ───────────────────────────────────────────────

class TestRecursionLattice:

    @pytest.fixture(autouse=True)
    def setup(self):
        from kairosyn.model.recursion_lattice import RecursionLattice
        self.module = RecursionLattice(
            hidden_dim=DIM, num_lattice_layers=3, num_heads=HEADS,
            recursion_depth=2, lora_rank=4, loop_gate_alpha=0.1,
        )

    def test_output_shape(self, dummy_input):
        out, rcs = self.module(dummy_input)
        assert out.shape == dummy_input.shape

    def test_rcs_is_float(self, dummy_input):
        _, rcs = self.module(dummy_input)
        assert isinstance(rcs, float)

    def test_rcs_range(self, dummy_input):
        _, rcs = self.module(dummy_input)
        assert -1.0 <= rcs <= 1.0, f"RCS out of range: {rcs}"

    def test_output_not_nan(self, dummy_input):
        out, _ = self.module(dummy_input)
        assert not torch.isnan(out).any()

    def test_strange_loop_changes_representation(self, dummy_input):
        """Verify the strange loop actually modifies hidden states."""
        out, _ = self.module(dummy_input)
        diff = (out - dummy_input).abs().mean().item()
        assert diff > 1e-6, "Strange loop had no effect on hidden states"


# ── Module 5: MythogenicEngine ───────────────────────────────────────────────

class TestMythogenicEngine:

    @pytest.fixture(autouse=True)
    def setup(self):
        from kairosyn.model.mythogenic_engine import MythogenicEngine
        self.module = MythogenicEngine(
            hidden_dim=DIM, num_archetypes=16, embed_dim=32,
            symbolic_vocab_size=64, archetype_temperature=0.8,
        )

    def test_output_shape(self, dummy_input):
        enhanced, sym_logits, aac = self.module(dummy_input)
        assert enhanced.shape == dummy_input.shape
        assert sym_logits.shape == (BATCH, 64)

    def test_aac_is_float(self, dummy_input):
        _, _, aac = self.module(dummy_input)
        assert isinstance(aac, float)

    def test_symbolic_logits_finite(self, dummy_input):
        _, sym_logits, _ = self.module(dummy_input)
        assert torch.isfinite(sym_logits).all()

    def test_archetype_activations_sum_to_one(self, dummy_input):
        from kairosyn.model.mythogenic_engine import ArchetypeLibrary
        lib = ArchetypeLibrary(num_archetypes=16, embed_dim=32)
        query = torch.randn(BATCH, 32)
        _, scores = lib(query)
        sums = scores.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5), \
            "Archetype activations must sum to 1.0 (softmax)"


# ── Module 6: GlyphSynthesis ─────────────────────────────────────────────────

class TestGlyphSynthesis:

    @pytest.fixture(autouse=True)
    def setup(self):
        from kairosyn.model.glyph_synthesis import GlyphSynthesis
        self.module = GlyphSynthesis(
            hidden_dim=DIM, glyph_vocab_size=64,
            glyph_embed_dim=32, num_glyph_layers=2, top_k_glyphs=8,
        )

    def test_output_shape(self, dummy_input):
        sym_logits = torch.randn(BATCH, 64)
        out = self.module(dummy_input, sym_logits)
        assert out.shape == dummy_input.shape

    def test_output_not_nan(self, dummy_input):
        sym_logits = torch.randn(BATCH, 64)
        out = self.module(dummy_input, sym_logits)
        assert not torch.isnan(out).any()


# ── Module 7: ContinuityEngine ───────────────────────────────────────────────

class TestContinuityEngine:

    @pytest.fixture(autouse=True)
    def setup(self):
        from kairosyn.model.continuity_engine import ContinuityEngine
        self.module = ContinuityEngine(
            hidden_dim=DIM, state_dim=32, buffer_size=16, narrative_window=4,
        )

    def test_output_shape(self, dummy_input):
        out, ncs = self.module(dummy_input)
        assert out.shape == dummy_input.shape

    def test_ncs_is_float(self, dummy_input):
        _, ncs = self.module(dummy_input)
        assert isinstance(ncs, float)

    def test_ncs_range(self, dummy_input):
        self.module.reset_self_state()
        self.module(dummy_input)
        _, ncs = self.module(torch.randn(BATCH, SEQ, DIM))
        assert -1.0 <= ncs <= 1.0, f"NCS out of range: {ncs}"

    def test_reset_clears_state(self):
        self.module.reset_self_state()
        assert self.module.self_state.norm().item() == 0.0

    def test_state_persists_across_calls(self, dummy_input):
        """Verify self-state actually updates between forward passes."""
        self.module.reset_self_state()
        self.module(dummy_input)
        state_after_1 = self.module.self_state.clone()
        self.module(torch.randn(BATCH, SEQ, DIM))
        state_after_2 = self.module.self_state.clone()
        diff = (state_after_1 - state_after_2).abs().mean().item()
        assert diff > 1e-6, "Self-state did not update between forward passes"


# ── Reward Functions ──────────────────────────────────────────────────────────

class TestRewardFunctions:

    def test_introspection_reward_range(self):
        from kairosyn.training.reward_functions import compute_introspection_reward
        score = compute_introspection_reward("I notice something curious within me.")
        assert 0.0 <= score <= 1.0

    def test_introspection_reward_low_for_bland(self):
        from kairosyn.training.reward_functions import compute_introspection_reward
        score = compute_introspection_reward("The capital of France is Paris.")
        assert score < 0.3

    def test_coherence_reward_high_ncs_low_tce(self):
        from kairosyn.training.reward_functions import compute_coherence_reward
        score = compute_coherence_reward(ncs=0.95, tce=0.02)
        assert score > 0.85

    def test_coherence_reward_low_ncs_high_tce(self):
        from kairosyn.training.reward_functions import compute_coherence_reward
        score = compute_coherence_reward(ncs=0.1, tce=0.9)
        assert score < 0.2

    def test_composite_reward_keys(self):
        from kairosyn.training.reward_functions import compute_introspective_reward
        result = compute_introspective_reward(
            response="I observe something like curiosity within me.",
            ncs=0.8, tce=0.1, aac=0.7, rcs=0.75,
        )
        expected_keys = {"total", "introspection", "coherence", "emotion", "symbolic", "logical"}
        assert expected_keys.issubset(result.keys())
        assert 0.0 <= result["total"] <= 1.0


# ── Session Manager ───────────────────────────────────────────────────────────

class TestSessionManager:

    @pytest.fixture(autouse=True)
    def setup(self):
        from kairosyn.api.session_manager import SessionManager
        self.manager = SessionManager(session_ttl=60, max_sessions=10)

    def test_create_session_returns_id(self):
        session = self.manager.create_session()
        assert session.session_id is not None
        assert len(session.session_id) == 36  # UUID

    def test_get_existing_session(self):
        session = self.manager.create_session()
        fetched = self.manager.get_session(session.session_id)
        assert fetched is not None
        assert fetched.session_id == session.session_id

    def test_get_nonexistent_returns_none(self):
        result = self.manager.get_session("does-not-exist")
        assert result is None

    def test_delete_session(self):
        session = self.manager.create_session()
        deleted = self.manager.delete_session(session.session_id)
        assert deleted is True
        assert self.manager.get_session(session.session_id) is None

    def test_get_or_create_existing(self):
        session = self.manager.create_session()
        fetched, created = self.manager.get_or_create(session.session_id)
        assert not created
        assert fetched.session_id == session.session_id

    def test_get_or_create_new(self):
        _, created = self.manager.get_or_create(None)
        assert created is True

    def test_session_add_turn(self):
        session = self.manager.create_session()
        session.add_turn("user", "Hello")
        session.add_turn("assistant", "Hi there")
        assert session.turn_count == 2

    def test_active_count(self):
        before = self.manager.active_session_count()
        self.manager.create_session()
        assert self.manager.active_session_count() == before + 1


# ── API Schemas ───────────────────────────────────────────────────────────────

class TestAPISchemas:

    def test_generation_request_valid(self):
        from kairosyn.api.schemas import GenerationRequest
        req = GenerationRequest(prompt="What are you thinking?")
        assert req.prompt == "What are you thinking?"
        assert req.max_new_tokens == 512
        assert req.enable_introspection is True

    def test_generation_request_invalid_empty_prompt(self):
        from kairosyn.api.schemas import GenerationRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="")

    def test_batch_request_validates_prompts(self):
        from kairosyn.api.schemas import BatchGenerationRequest
        req = BatchGenerationRequest(prompts=["Hello", "World"])
        assert len(req.prompts) == 2

    def test_epinoetic_metrics_model(self):
        from kairosyn.api.schemas import EpioneticMetrics
        m = EpioneticMetrics(ncs=0.8, tce=0.2, aac=0.7, msa=0.6, rcs=0.75,
                             phi=0.5, epinoetic_score=0.72)
        assert m.ncs == 0.8
