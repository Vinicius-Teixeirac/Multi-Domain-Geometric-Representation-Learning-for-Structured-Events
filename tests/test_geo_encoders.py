"""Tests for the geo (WHERE) domain encoders: construction, forward shapes,
the sphere-output invariant for Riemannian variants, and the build_geo_encoder
factory."""

import torch
import pytest

from src.models.multi_domain.geo_encoders import (
    HypersphericalEncoder,
    ProjectedEncoder,
    EuclideanEncoder,
    RegionAwareEncoder,
    build_geo_encoder,
)

BATCH_SIZE = 8
OUT_DIM = 16
HIDDEN_DIM = 32
REGION_CARDINALITY = 10


@pytest.fixture
def lat_lon() -> torch.Tensor:
    """Random valid (lat, lon) pairs in degrees, shape (BATCH_SIZE, 2)."""
    lat = torch.empty(BATCH_SIZE, 1).uniform_(-90, 90)
    lon = torch.empty(BATCH_SIZE, 1).uniform_(-180, 180)
    return torch.cat([lat, lon], dim=-1)


@pytest.fixture
def geo_country_idx() -> torch.Tensor:
    """Random country indices in [0, REGION_CARDINALITY), shape (BATCH_SIZE,)."""
    return torch.randint(0, REGION_CARDINALITY, (BATCH_SIZE,))


@pytest.mark.parametrize(
    "encoder_cls,lives_on_sphere",
    [
        (HypersphericalEncoder, True),
        (ProjectedEncoder, True),
        (EuclideanEncoder, False),
    ],
)
class TestGeoEncoders:
    def test_forward_shape(self, encoder_cls, lives_on_sphere, lat_lon):
        """Every geo encoder maps (B, 2) lat/lon degrees to (B, out_dim)."""
        enc = encoder_cls(out_dim=OUT_DIM, hidden_dim=HIDDEN_DIM)
        out = enc(lat_lon)
        assert out.shape == (BATCH_SIZE, OUT_DIM)
        assert out.dtype == torch.float32

    def test_unit_norm_when_on_sphere(self, encoder_cls, lives_on_sphere, lat_lon):
        """Hyperspherical/projected outputs live on S^{out_dim-1}; Euclidean does not."""
        enc = encoder_cls(out_dim=OUT_DIM, hidden_dim=HIDDEN_DIM)
        out = enc(lat_lon)
        norms = out.norm(dim=-1)
        if lives_on_sphere:
            assert torch.allclose(norms, torch.ones(BATCH_SIZE), atol=1e-4)
        else:
            assert not torch.allclose(norms, torch.ones(BATCH_SIZE), atol=1e-4)

    def test_geo_country_idx_argument_is_ignored(self, encoder_cls, lives_on_sphere, lat_lon, geo_country_idx):
        """These encoders accept geo_country_idx for API uniformity but never use it."""
        enc = encoder_cls(out_dim=OUT_DIM, hidden_dim=HIDDEN_DIM)
        out_with = enc(lat_lon, geo_country_idx)
        out_without = enc(lat_lon, None)
        assert torch.allclose(out_with, out_without, atol=1e-6)


class TestRegionAwareEncoder:
    def test_forward_shape_and_unit_norm(self, lat_lon, geo_country_idx):
        """RegionAwareEncoder blends a country embedding but still outputs a point on S^{out-1}."""
        enc = RegionAwareEncoder(
            out_dim=OUT_DIM, hidden_dim=HIDDEN_DIM,
            region_cardinality=REGION_CARDINALITY, region_embed_dim=8,
        )
        out = enc(lat_lon, geo_country_idx)
        assert out.shape == (BATCH_SIZE, OUT_DIM)
        assert torch.allclose(out.norm(dim=-1), torch.ones(BATCH_SIZE), atol=1e-4)

    def test_none_country_idx_uses_zero_embedding(self, lat_lon):
        """Passing geo_country_idx=None must not crash; it falls back to the zero (unknown) embedding."""
        enc = RegionAwareEncoder(
            out_dim=OUT_DIM, hidden_dim=HIDDEN_DIM,
            region_cardinality=REGION_CARDINALITY, region_embed_dim=8,
        )
        out = enc(lat_lon, None)
        assert out.shape == (BATCH_SIZE, OUT_DIM)
        assert not torch.isnan(out).any()

    def test_different_countries_give_different_embeddings(self, lat_lon):
        """Two distinct non-zero country indices should produce distinct outputs (embedding actually used)."""
        enc = RegionAwareEncoder(
            out_dim=OUT_DIM, hidden_dim=HIDDEN_DIM,
            region_cardinality=REGION_CARDINALITY, region_embed_dim=8,
        )
        idx_a = torch.ones(BATCH_SIZE, dtype=torch.long)
        idx_b = torch.full((BATCH_SIZE,), 2, dtype=torch.long)
        out_a = enc(lat_lon, idx_a)
        out_b = enc(lat_lon, idx_b)
        assert not torch.allclose(out_a, out_b, atol=1e-5)


class TestBuildGeoEncoder:
    @pytest.mark.parametrize(
        "type_name,expected_cls",
        [
            ("hyperspherical", HypersphericalEncoder),
            ("projected", ProjectedEncoder),
            ("euclidean", EuclideanEncoder),
            ("region_aware", RegionAwareEncoder),
        ],
    )
    def test_dispatches_to_correct_class(self, type_name, expected_cls):
        """build_geo_encoder must instantiate the class registered for each known type string."""
        cfg = {"type": type_name, "out_dim": OUT_DIM, "hidden_dim": HIDDEN_DIM}
        enc = build_geo_encoder(cfg, region_cardinality=REGION_CARDINALITY)
        assert isinstance(enc, expected_cls)

    def test_defaults_to_hyperspherical(self):
        """Omitting 'type' should default to the truly Riemannian hyperspherical encoder."""
        enc = build_geo_encoder({})
        assert isinstance(enc, HypersphericalEncoder)

    def test_unknown_type_raises(self):
        """An unrecognized encoder type must raise ValueError, not silently pick a default."""
        with pytest.raises(ValueError, match="Unknown geo encoder type"):
            build_geo_encoder({"type": "not_a_real_type"})
