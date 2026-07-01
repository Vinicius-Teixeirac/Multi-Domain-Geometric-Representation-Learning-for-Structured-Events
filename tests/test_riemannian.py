"""Tests for the shared Riemannian primitives (log/exp maps, SphericalLinear,
SphereReLU) that every multi-domain hyperspherical encoder builds on."""

import torch
import torch.nn.functional as F
import pytest

from src.models.multi_domain.riemannian import log_north, exp_north, SphericalLinear, SphereReLU

BATCH_SIZE = 8


def _random_sphere_points(n: int, dim: int) -> torch.Tensor:
    """Return n random unit vectors on S^{dim-1} (uniform direction, not near the north pole)."""
    x = torch.randn(n, dim)
    return F.normalize(x, p=2, dim=-1)


@pytest.fixture
def sphere_points() -> torch.Tensor:
    """Random points on S^2 (shape (BATCH_SIZE, 3))."""
    return _random_sphere_points(BATCH_SIZE, 3)


class TestLogExpNorth:
    def test_log_north_last_coord_is_zero(self, sphere_points):
        """log_north always returns a tangent vector with an exact zero last coordinate."""
        v = log_north(sphere_points)
        assert torch.allclose(v[..., -1], torch.zeros(BATCH_SIZE), atol=1e-6)

    def test_north_pole_maps_to_zero_tangent(self):
        """The north pole itself is the reference point, so its log map is the zero vector."""
        d = 4
        north_pole = torch.zeros(1, d)
        north_pole[0, -1] = 1.0
        v = log_north(north_pole)
        assert torch.allclose(v, torch.zeros(1, d), atol=1e-5)

    def test_exp_of_zero_is_north_pole(self):
        """exp_north(0) must return exactly the north pole (inverse of the above)."""
        d = 5
        v = torch.zeros(1, d)
        x = exp_north(v)
        expected = torch.zeros(1, d)
        expected[0, -1] = 1.0
        assert torch.allclose(x, expected, atol=1e-5)

    def test_log_exp_round_trip(self, sphere_points):
        """exp_north(log_north(x)) reconstructs x for generic points on the sphere."""
        v = log_north(sphere_points)
        x_reconstructed = exp_north(v)
        assert torch.allclose(x_reconstructed, sphere_points, atol=1e-4)

    def test_exp_output_has_unit_norm(self, sphere_points):
        """exp_north always lands back on the unit sphere, regardless of input tangent vector."""
        v = log_north(sphere_points) * 2.0  # scale to a different tangent vector
        x = exp_north(v)
        norms = x.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(BATCH_SIZE), atol=1e-4)


class TestSphericalLinear:
    def test_output_shape_and_unit_norm(self, sphere_points):
        """SphericalLinear maps S^{in-1} -> S^{out-1}; output must have unit norm."""
        layer = SphericalLinear(in_dim=3, out_dim=6)
        out = layer(sphere_points)
        assert out.shape == (BATCH_SIZE, 6)
        assert torch.allclose(out.norm(dim=-1), torch.ones(BATCH_SIZE), atol=1e-4)

    def test_output_dim_one_is_degenerate_but_stable(self, sphere_points):
        """out_dim=1 means S^0 = {-1, +1}; the layer should still run without NaNs."""
        layer = SphericalLinear(in_dim=3, out_dim=1)
        out = layer(sphere_points)
        assert out.shape == (BATCH_SIZE, 1)
        assert not torch.isnan(out).any()


class TestSphereReLU:
    def test_output_stays_on_sphere(self, sphere_points):
        """SphereReLU is applied in tangent space but must return a point back on S^{d-1}."""
        act = SphereReLU()
        out = act(sphere_points)
        assert out.shape == sphere_points.shape
        assert torch.allclose(out.norm(dim=-1), torch.ones(BATCH_SIZE), atol=1e-4)

    def test_north_pole_is_a_fixed_point(self):
        """ReLU(0) = 0, so the north pole (zero tangent vector) must map to itself."""
        d = 4
        north_pole = torch.zeros(1, d)
        north_pole[0, -1] = 1.0
        act = SphereReLU()
        out = act(north_pole)
        assert torch.allclose(out, north_pole, atol=1e-5)
