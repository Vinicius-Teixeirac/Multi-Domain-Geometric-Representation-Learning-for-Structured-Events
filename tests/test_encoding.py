"""Tests for the tabular column encoders: SafeLabelEncoder, HashEncoder, HashedOneHotEncoder."""

import pandas as pd
import pytest

from src.representation.tabular.encoding import SafeLabelEncoder, HashEncoder, HashedOneHotEncoder


class TestSafeLabelEncoder:
    def test_fit_assigns_contiguous_ids_starting_at_one(self):
        """Known categories start at 1; 0 is reserved for UNK."""
        enc = SafeLabelEncoder().fit(pd.Series(["a", "b", "c"]))
        assert enc.num_classes_ == 3
        assert set(enc.mapping.values()) == {1, 2, 3}
        assert 0 not in enc.mapping.values()

    def test_transform_maps_known_values(self):
        """A fitted encoder must round-trip known categories to their assigned integer codes."""
        enc = SafeLabelEncoder().fit(pd.Series(["a", "b"]))
        out = enc.transform(pd.Series(["a", "b", "a"]))
        assert out.tolist() == [enc.mapping["a"], enc.mapping["b"], enc.mapping["a"]]

    def test_unseen_value_maps_to_unk(self):
        """A category never seen during fit must map to the UNK token (0), not raise."""
        enc = SafeLabelEncoder().fit(pd.Series(["a", "b"]))
        out = enc.transform(pd.Series(["a", "never_seen"]))
        assert out.iloc[1] == 0

    def test_transform_before_fit_raises(self):
        """Calling transform before fit is a programming error and must raise."""
        enc = SafeLabelEncoder()
        with pytest.raises(RuntimeError):
            enc.transform(pd.Series(["a"]))

    def test_double_fit_raises(self):
        """Refitting an already-fitted encoder must raise, not silently overwrite the mapping."""
        enc = SafeLabelEncoder().fit(pd.Series(["a"]))
        with pytest.raises(RuntimeError):
            enc.fit(pd.Series(["b"]))

    def test_save_load_round_trip(self, tmp_path):
        """A saved encoder, reloaded, must transform identically to the original."""
        enc = SafeLabelEncoder().fit(pd.Series(["a", "b", "c"]))
        path = tmp_path / "enc.json"
        enc.save(path)
        loaded = SafeLabelEncoder.load(path)

        original = enc.transform(pd.Series(["a", "b", "unseen"]))
        restored = loaded.transform(pd.Series(["a", "b", "unseen"]))
        assert original.tolist() == restored.tolist()


class TestHashEncoder:
    def test_transform_is_within_bucket_range(self):
        """All hash outputs must fall in [0, num_buckets)."""
        enc = HashEncoder(num_buckets=16)
        out = enc.transform(pd.Series([f"value_{i}" for i in range(50)]))
        assert out.between(0, 15).all()

    def test_transform_is_deterministic(self):
        """The same input value must always hash to the same bucket."""
        enc = HashEncoder(num_buckets=1024)
        s = pd.Series(["repeat_me"] * 5)
        out = enc.transform(s)
        assert out.nunique() == 1

    def test_nan_treated_as_dedicated_string(self):
        """NaN values must hash consistently (treated as the literal string '__nan__')."""
        enc = HashEncoder(num_buckets=1024)
        out = enc.transform(pd.Series([None, float("nan")]))
        assert out.iloc[0] == out.iloc[1]

    def test_is_stateless_fit_is_noop(self):
        """fit() must be a no-op returning self, since HashEncoder needs no training."""
        enc = HashEncoder(num_buckets=16)
        assert enc.fit(pd.Series(["a"])) is enc
        assert enc.is_fitted is True

    def test_save_load_round_trip(self, tmp_path):
        """A saved encoder, reloaded, must hash identically to the original."""
        enc = HashEncoder(num_buckets=256)
        path = tmp_path / "hash.json"
        enc.save(path)
        loaded = HashEncoder.load(path)

        s = pd.Series(["x", "y", "z"])
        assert enc.transform(s).tolist() == loaded.transform(s).tolist()


class TestHashedOneHotEncoder:
    def test_output_shape(self):
        """Output must be a sparse (N, num_buckets) matrix."""
        enc = HashedOneHotEncoder(num_buckets=32)
        out = enc.transform(pd.Series(["a", "b", "c"]))
        assert out.shape == (3, 32)

    def test_signed_values_are_plus_or_minus_one(self):
        """With signed=True, every nonzero entry must be exactly +1 or -1."""
        enc = HashedOneHotEncoder(num_buckets=32, signed=True)
        out = enc.transform(pd.Series([f"v{i}" for i in range(20)])).toarray()
        nonzero = out[out != 0]
        assert set(nonzero.tolist()) <= {1.0, -1.0}

    def test_unsigned_values_are_one(self):
        """With signed=False, every nonzero entry must be exactly +1 (plain one-hot)."""
        enc = HashedOneHotEncoder(num_buckets=32, signed=False)
        out = enc.transform(pd.Series([f"v{i}" for i in range(20)])).toarray()
        nonzero = out[out != 0]
        assert set(nonzero.tolist()) <= {1.0}
