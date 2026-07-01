"""Tests for the temporal (WHEN) domain encoders: construction, forward shapes,
the sphere-output invariant for the Riemannian variant, learnable-period
behavior, and the build_temporal_encoder factory."""

import torch
import pytest

from src.models.multi_domain.temporal_encoders import (
    ProductManifoldEncoder,
    LearnablePeriodEncoder,
    FourierEncoder,
    RiemannianProductEncoder,
    build_temporal_encoder,
)

BATCH_SIZE = 8
OUT_DIM = 12
HIDDEN_DIM = 24


@pytest.fixture
def time_features() -> torch.Tensor:
    """Random (t_linear, doy, dow) triples, shape (BATCH_SIZE, 3), matching the
    dataset's raw layout: t_linear ~ N(0, 1), doy in [1, 366], dow in [0, 6]."""
    t_linear = torch.randn(BATCH_SIZE, 1)
    doy = torch.empty(BATCH_SIZE, 1).uniform_(1, 366)
    dow = torch.empty(BATCH_SIZE, 1).uniform_(0, 6)
    return torch.cat([t_linear, doy, dow], dim=-1)


@pytest.mark.parametrize(
    "encoder_cls",
    [ProductManifoldEncoder, LearnablePeriodEncoder, FourierEncoder, RiemannianProductEncoder],
)
class TestTemporalEncoders:
    def test_forward_shape(self, encoder_cls, time_features):
        """Every temporal encoder maps (B, 3) raw time features to (B, out_dim)."""
        enc = encoder_cls(out_dim=OUT_DIM, hidden_dim=HIDDEN_DIM)
        out = enc(time_features)
        assert out.shape == (BATCH_SIZE, OUT_DIM)
        assert out.dtype == torch.float32
        assert not torch.isnan(out).any()


class TestRiemannianProductEncoderSphere:
    def test_output_has_unit_norm(self, time_features):
        """Unlike the other three (Euclidean MLP output), this encoder's output must live on S^{out-1}."""
        enc = RiemannianProductEncoder(out_dim=OUT_DIM, hidden_dim=HIDDEN_DIM)
        out = enc(time_features)
        assert torch.allclose(out.norm(dim=-1), torch.ones(BATCH_SIZE), atol=1e-4)


class TestLearnablePeriodEncoder:
    def test_periods_initialised_near_calendar_values(self):
        """Log-space periods should start at 365 (annual) and 7 (weekly) before any training."""
        enc = LearnablePeriodEncoder(out_dim=OUT_DIM, hidden_dim=HIDDEN_DIM)
        assert torch.isclose(enc.log_period_annual.exp(), torch.tensor(365.0), atol=1e-3)
        assert torch.isclose(enc.log_period_weekly.exp(), torch.tensor(7.0), atol=1e-3)

    def test_periods_are_learnable_parameters(self):
        """The periods must be nn.Parameters (requires_grad) so training can adapt them."""
        enc = LearnablePeriodEncoder(out_dim=OUT_DIM, hidden_dim=HIDDEN_DIM)
        assert enc.log_period_annual.requires_grad
        assert enc.log_period_weekly.requires_grad

    def test_periods_stay_positive_after_a_gradient_step(self, time_features):
        """Log-space parameterisation must keep periods > 0 even after a large gradient update.

        Note: the step size is kept moderate (not huge) on purpose -- driving
        log_period far enough negative makes exp() underflow to exactly 0.0 in
        float32, which would make this test's own assertion unreliable rather
        than exercising the parameterisation.
        """
        enc = LearnablePeriodEncoder(out_dim=OUT_DIM, hidden_dim=HIDDEN_DIM)
        out = enc(time_features)
        out.sum().backward()
        with torch.no_grad():
            enc.log_period_annual -= 5.0 * enc.log_period_annual.grad
            enc.log_period_weekly -= 5.0 * enc.log_period_weekly.grad
        assert enc.log_period_annual.exp().item() > 0
        assert enc.log_period_weekly.exp().item() > 0


class TestBuildTemporalEncoder:
    @pytest.mark.parametrize(
        "type_name,expected_cls",
        [
            ("product_manifold", ProductManifoldEncoder),
            ("learnable_period", LearnablePeriodEncoder),
            ("fourier", FourierEncoder),
            ("riemannian_product", RiemannianProductEncoder),
        ],
    )
    def test_dispatches_to_correct_class(self, type_name, expected_cls):
        """build_temporal_encoder must instantiate the class registered for each known type string."""
        cfg = {"type": type_name, "out_dim": OUT_DIM, "hidden_dim": HIDDEN_DIM}
        enc = build_temporal_encoder(cfg)
        assert isinstance(enc, expected_cls)

    def test_defaults_to_product_manifold(self):
        """Omitting 'type' should default to the fixed-period product manifold encoder."""
        enc = build_temporal_encoder({})
        assert isinstance(enc, ProductManifoldEncoder)

    def test_fourier_num_frequencies_is_configurable(self):
        """num_frequencies should control the learnable frequency matrix width."""
        cfg = {"type": "fourier", "out_dim": OUT_DIM, "hidden_dim": HIDDEN_DIM, "num_frequencies": 4}
        enc = build_temporal_encoder(cfg)
        assert enc.W.shape == (4, 3)

    def test_unknown_type_raises(self):
        """An unrecognized encoder type must raise ValueError, not silently pick a default."""
        with pytest.raises(ValueError, match="Unknown temporal encoder type"):
            build_temporal_encoder({"type": "not_a_real_type"})
