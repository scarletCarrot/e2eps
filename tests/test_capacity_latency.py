import numpy as np

from e2eps.capacity import CapacityAllocator, gather_serving
from e2eps.config.schema import LatencyConfig
from e2eps.constants import C_LIGHT
from e2eps.geometry import GeometryResult
from e2eps.latency import LatencyModel
from e2eps.rf import LinkResult


def _geo(el, rng):
    el = np.asarray(el, dtype=np.float32)[None]          # (1, S, G)
    rng = np.asarray(rng, dtype=np.float32)[None]
    return GeometryResult(el, rng, np.zeros_like(el), el >= 25)


def _link(rate):
    rate = np.asarray(rate, dtype=np.float32)[None]
    avail = rate > 0
    return LinkResult(np.where(avail, 10.0, np.nan).astype(np.float32),
                      np.where(avail, 5, -1).astype(np.int16),
                      np.where(avail, 2.0, 0.0).astype(np.float32), rate, avail)


# 3 satellites x 3 terminals
EL = [[60, 30, 10],
      [40, 70, 10],
      [10, 10, 10]]
RATE = [[900e6, 600e6, 0],
        [700e6, 950e6, 0],
        [0, 0, 0]]
RNG = [[6e5, 9e5, 2e6],
       [8e5, 6e5, 2e6],
       [2e6, 2e6, 2e6]]


def test_highest_elevation_and_equal_share():
    a = CapacityAllocator().allocate(_geo(EL, RNG), _link(RATE), np.ones((1, 3), bool))
    assert a.serving_sat[0].tolist() == [0, 1, -1]
    assert a.users_per_sat[0].tolist() == [1, 1, 0]
    assert a.throughput_bps[0].tolist() == [900e6, 950e6, 0]


def test_shared_satellite_splits_capacity():
    el = [[60, 50, 10], [40, 30, 10], [10, 10, 10]]      # both terminals prefer sat 0
    a = CapacityAllocator().allocate(_geo(el, RNG), _link(RATE), np.ones((1, 3), bool))
    assert a.serving_sat[0].tolist() == [0, 0, -1]
    assert a.users_per_sat[0, 0] == 2
    assert np.allclose(a.throughput_bps[0, :2], [450e6, 300e6])


def test_no_gateway_means_no_service():
    has_gw = np.array([[False, True, True]])
    a = CapacityAllocator().allocate(_geo(EL, RNG), _link(RATE), has_gw)
    assert a.serving_sat[0].tolist() == [1, 1, -1]      # falls back to sat 1


def test_gather_serving():
    arr = np.arange(9, dtype=float).reshape(1, 3, 3)
    out = gather_serving(arr, np.array([[2, 0, -1]]))
    assert out[0, 0] == arr[0, 2, 0] and out[0, 1] == arr[0, 0, 1] and np.isnan(out[0, 2])


def test_latency_bent_pipe():
    geo_ut = _geo(EL, RNG)
    gw_geo = GeometryResult(np.full((1, 3, 1), 40, np.float32),
                            np.array([[[7e5], [1e6], [3e6]]], np.float32),
                            np.zeros((1, 3, 1), np.float32),
                            np.array([[[True], [True], [False]]]))
    model = LatencyModel(LatencyConfig(processing_ms=5, terrestrial_ms=10))
    feeder = model.feeder_range(gw_geo)
    assert np.isinf(feeder[0, 2])
    lat = model.one_way_ms(geo_ut, np.array([[0, 1, -1]]), feeder)
    assert np.isclose(lat[0, 0], (6e5 + 7e5) / C_LIGHT * 1e3 + 15)
    assert np.isclose(lat[0, 1], (6e5 + 1e6) / C_LIGHT * 1e3 + 15)
    assert np.isnan(lat[0, 2])
