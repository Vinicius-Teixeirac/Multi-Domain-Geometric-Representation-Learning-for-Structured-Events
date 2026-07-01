"""Tests for the tabular value transformers: date parsing/cyclic encoding, geo
Cartesian conversion, and the StandardScaler wrapper."""

import numpy as np
import pandas as pd
import pytest

from src.representation.tabular.transformers import (
    IntegerDateParser,
    DateToCyclic,
    GeoToCartesian,
    StandardScalerWrapper,
)


class TestIntegerDateParser:
    def test_parses_yyyymmdd(self):
        """A GDELT-style integer date (e.g. 20150923) must parse to the correct datetime."""
        parser = IntegerDateParser(fmt="%Y%m%d")
        out = parser.transform(pd.Series([20150923, 20200101]))
        assert out.iloc[0] == pd.Timestamp("2015-09-23")
        assert out.iloc[1] == pd.Timestamp("2020-01-01")


class TestDateToCyclic:
    def test_output_columns_named_from_series(self):
        """Output columns must be named '{series_name}_sin'/'{series_name}_cos'."""
        s = pd.Series([1, 100, 365], name="doy")
        out = DateToCyclic(period=365).transform(s)
        assert set(out.columns) == {"doy_sin", "doy_cos"}

    def test_period_boundary_wraps_to_same_point(self):
        """Day 0 and day == period should map to (nearly) the same point on the circle."""
        out = DateToCyclic(period=365).transform(pd.Series([0, 365], name="doy"))
        assert out["doy_sin"].iloc[0] == pytest.approx(out["doy_sin"].iloc[1], abs=1e-6)
        assert out["doy_cos"].iloc[0] == pytest.approx(out["doy_cos"].iloc[1], abs=1e-6)

    def test_quarter_period_gives_unit_sin(self):
        """At exactly 1/4 of the period, sin should be 1 and cos should be 0."""
        out = DateToCyclic(period=100).transform(pd.Series([25], name="x"))
        assert out["x_sin"].iloc[0] == pytest.approx(1.0, abs=1e-6)
        assert out["x_cos"].iloc[0] == pytest.approx(0.0, abs=1e-6)


class TestGeoToCartesian:
    def test_equator_prime_meridian_is_unit_x(self):
        """(lat=0, lon=0) on Earth must map to the point (1, 0, 0) on the unit sphere."""
        df = pd.DataFrame({"lat": [0.0], "lon": [0.0]})
        out = GeoToCartesian(prefix="loc").transform(df)
        assert out["loc_geo_x"].iloc[0] == pytest.approx(1.0, abs=1e-6)
        assert out["loc_geo_y"].iloc[0] == pytest.approx(0.0, abs=1e-6)
        assert out["loc_geo_z"].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_north_pole_is_unit_z(self):
        """(lat=90, lon=anything) must map to (0, 0, 1) regardless of longitude."""
        df = pd.DataFrame({"lat": [90.0], "lon": [123.0]})
        out = GeoToCartesian(prefix="loc").transform(df)
        assert out["loc_geo_x"].iloc[0] == pytest.approx(0.0, abs=1e-6)
        assert out["loc_geo_y"].iloc[0] == pytest.approx(0.0, abs=1e-6)
        assert out["loc_geo_z"].iloc[0] == pytest.approx(1.0, abs=1e-6)

    def test_output_is_always_unit_norm(self):
        """Every output row must lie on the unit sphere (x^2 + y^2 + z^2 == 1), regardless of input."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "lat": rng.uniform(-90, 90, size=20),
            "lon": rng.uniform(-180, 180, size=20),
        })
        out = GeoToCartesian(prefix="loc").transform(df)
        norms = np.sqrt(out["loc_geo_x"] ** 2 + out["loc_geo_y"] ** 2 + out["loc_geo_z"] ** 2)
        assert np.allclose(norms, 1.0, atol=1e-6)


class TestStandardScalerWrapper:
    def test_transform_has_zero_mean_unit_variance(self):
        """After fitting on the same data it transforms, output must be standardised."""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        scaler = StandardScalerWrapper().fit(s)
        out = scaler.transform(s)
        assert out.mean() == pytest.approx(0.0, abs=1e-6)
        assert out.std(ddof=0) == pytest.approx(1.0, abs=1e-6)

    def test_save_load_round_trip(self, tmp_path):
        """A saved scaler, reloaded, must transform identically to the original."""
        s = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0])
        scaler = StandardScalerWrapper().fit(s)
        path = tmp_path / "scaler.json"
        scaler.save(path)
        loaded = StandardScalerWrapper.load(path)

        new_data = pd.Series([5.0, 6.0])
        pd.testing.assert_series_equal(
            scaler.transform(new_data).reset_index(drop=True),
            loaded.transform(new_data).reset_index(drop=True),
        )
