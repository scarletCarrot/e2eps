import numpy as np

from e2eps.config.schema import SimulationConfig
from e2eps.timegrid import iter_chunks, make_time_grid


def test_time_grid_and_chunks_cover_all_steps():
    sim = SimulationConfig(duration_s=600, step_s=60, chunk_steps=4)
    t = make_time_grid(sim)
    assert t.size == 11 and t[-1] == 600
    chunks = list(iter_chunks(t, sim.chunk_steps))
    assert [c[1].size for c in chunks] == [4, 4, 3]
    assert np.array_equal(np.concatenate([c[1] for c in chunks]), t)
