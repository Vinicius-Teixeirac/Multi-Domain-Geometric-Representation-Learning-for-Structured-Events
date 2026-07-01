"""Tests for the fusion mechanisms (Psi) that combine per-domain views into
class logits: construction, forward shapes, the geometry-aware log-map
dimension bookkeeping, and the build_fusion factory."""

import torch
import torch.nn.functional as F
import pytest

from src.models.multi_domain.fusion import (
    ConcatMLPFusion,
    AttentionFusion,
    GatedFusion,
    GeometryAwareFusion,
    build_fusion,
)

BATCH_SIZE = 8
NUM_CLASSES = 4
VIEW_DIMS = [16, 12, 8]  # e.g. actor_out, geo_out, temporal_out


@pytest.fixture
def views() -> list:
    """Three random per-domain view tensors matching VIEW_DIMS."""
    return [torch.randn(BATCH_SIZE, d) for d in VIEW_DIMS]


@pytest.mark.parametrize("fusion_cls", [ConcatMLPFusion, AttentionFusion, GatedFusion])
class TestBlindFusion:
    def test_forward_shape(self, fusion_cls, views):
        """Every geometry-blind fusion module reduces the view list to (B, num_classes) logits."""
        fusion = fusion_cls(view_dims=VIEW_DIMS, num_classes=NUM_CLASSES)
        out = fusion(views)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)
        assert not torch.isnan(out).any()


class TestGeometryAwareFusion:
    def _sphere_views(self):
        """Two sphere-valued views (unit norm) and one Euclidean view, matching VIEW_DIMS."""
        sphere_a = F.normalize(torch.randn(BATCH_SIZE, VIEW_DIMS[0]), dim=-1)
        sphere_b = F.normalize(torch.randn(BATCH_SIZE, VIEW_DIMS[1]), dim=-1)
        euclidean = torch.randn(BATCH_SIZE, VIEW_DIMS[2])
        return [sphere_a, sphere_b, euclidean]

    def test_forward_shape_via_build_fusion(self):
        """build_fusion must size the inner module to the *effective* (post-log-map) dims,
        i.e. sphere views lose one dimension each; this only works end-to-end if the
        inner module's Linear layers were built with the right input width."""
        views = self._sphere_views()
        view_manifolds = ["sphere", "sphere", "euclidean"]
        cfg = {"type": "geometry_aware", "inner_type": "concat_mlp"}
        fusion = build_fusion(cfg, VIEW_DIMS, NUM_CLASSES, view_manifolds=view_manifolds)
        assert isinstance(fusion, GeometryAwareFusion)
        out = fusion(views)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)
        assert not torch.isnan(out).any()

    def test_default_view_manifolds_all_euclidean(self):
        """build_fusion without view_manifolds should treat every view as Euclidean (no log map)."""
        views = [torch.randn(BATCH_SIZE, d) for d in VIEW_DIMS]
        cfg = {"type": "geometry_aware", "inner_type": "concat_mlp"}
        fusion = build_fusion(cfg, VIEW_DIMS, NUM_CLASSES)
        out = fusion(views)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    @pytest.mark.parametrize("inner_type", ["concat_mlp", "attention", "gated"])
    def test_all_inner_fusion_types_work(self, inner_type):
        """geometry_aware fusion must delegate correctly to each of the three inner fusion types."""
        views = self._sphere_views()
        view_manifolds = ["sphere", "sphere", "euclidean"]
        cfg = {"type": "geometry_aware", "inner_type": inner_type}
        fusion = build_fusion(cfg, VIEW_DIMS, NUM_CLASSES, view_manifolds=view_manifolds)
        out = fusion(views)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)


class TestBuildFusion:
    @pytest.mark.parametrize(
        "type_name,expected_cls",
        [
            ("concat_mlp", ConcatMLPFusion),
            ("attention", AttentionFusion),
            ("gated", GatedFusion),
        ],
    )
    def test_blind_types_dispatch_to_correct_class(self, type_name, expected_cls, views):
        """build_fusion must instantiate the geometry-blind class registered for each type string."""
        cfg = {"type": type_name}
        fusion = build_fusion(cfg, VIEW_DIMS, NUM_CLASSES)
        assert isinstance(fusion, expected_cls)

    def test_defaults_to_concat_mlp(self, views):
        """Omitting 'type' should default to the simplest late-concatenation baseline."""
        fusion = build_fusion({}, VIEW_DIMS, NUM_CLASSES)
        assert isinstance(fusion, ConcatMLPFusion)

    def test_unknown_type_raises(self):
        """An unrecognized fusion type must raise ValueError, not silently pick a default."""
        with pytest.raises(ValueError, match="Unknown fusion type"):
            build_fusion({"type": "not_a_real_type"}, VIEW_DIMS, NUM_CLASSES)
