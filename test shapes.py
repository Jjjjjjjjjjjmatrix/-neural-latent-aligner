import pytest

torch = pytest.importorskip("torch")

from nla.losses import total_loss
from nla.models.nla import NeuralLatentAligner

B, E, T = 4, 64, 128


@pytest.fixture
def model():
    return NeuralLatentAligner(n_electrodes=E, latent_dim=64, content_dim=16, hidden_dim=64)


def test_forward_shapes(model):
    x, xp = torch.randn(B, E, T), torch.randn(B, E, T)
    out = model(x, xp)
    assert out["x_hat"].shape == x.shape
    assert out["c_x"].shape == out["c_aligned"].shape
    assert out["A"].shape[1] == out["A"].shape[2]


def test_alignment_rows_sum_to_one(model):
    out = model(torch.randn(B, E, T), torch.randn(B, E, T))
    assert torch.allclose(out["A"].sum(-1), torch.ones_like(out["A"].sum(-1)), atol=1e-5)


def test_loss_backward(model):
    x, xp = torch.randn(B, E, T), torch.randn(B, E, T)
    loss, logs = total_loss(x, model(x, xp))
    loss.backward()
    assert all(v == v for v in logs.values())  # pas de NaN
    assert any(p.grad is not None for p in model.parameters())


def test_content_bottleneck():
    with pytest.raises(AssertionError):
        NeuralLatentAligner(n_electrodes=E, latent_dim=32, content_dim=64)
