"""
KAIROSYN pytest configuration
==============================
Shared fixtures and markers for the test suite.
"""

import pytest
import torch


# ── Markers ───────────────────────────────────────────────────────────────────
def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: test requires CUDA GPU")
    config.addinivalue_line("markers", "slow: test takes > 30 seconds")
    config.addinivalue_line("markers", "integration: requires full model load")


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def hidden_dim():
    return 64


@pytest.fixture(scope="session")
def batch_size():
    return 2


@pytest.fixture(scope="session")
def seq_len():
    return 16


@pytest.fixture(scope="session")
def dummy_hidden(batch_size, seq_len, hidden_dim):
    """Standard CPU hidden state tensor for module testing."""
    torch.manual_seed(42)
    return torch.randn(batch_size, seq_len, hidden_dim)
