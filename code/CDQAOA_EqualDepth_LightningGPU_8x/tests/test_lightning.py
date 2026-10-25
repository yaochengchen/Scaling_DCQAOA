import numpy as np
import pytest
pytest.importorskip('pennylane')
from cdtransfer.instances import make_instance
from cdtransfer.lightning import LightningSimulator, LightningTapeSimulator
from cdtransfer.simulator import StatevectorSimulator, Angles
from cdtransfer.optimization import run_lbfgs, ramp_start


@pytest.mark.parametrize('cls', [LightningSimulator, LightningTapeSimulator])
@pytest.mark.parametrize('p', [1, 3])
@pytest.mark.parametrize('cd', [False, True])
def test_state_energy_gradient_match_numpy_reference(cls, p, cd):
    inst = make_instance(6, 'er05', 'signed_normal', 77)
    sim, ref = cls(inst, device='lightning.qubit'), StatevectorSimulator(inst)
    x = np.random.default_rng(19).uniform(-.3, .3, (3 if cd else 2) * p)
    if cd:
        x[2 * p] = 0.
    a = Angles.unpack(x, p, cd)
    value, g = sim.value_and_grad(x, p, cd)
    assert value == pytest.approx(ref.evaluate(a), abs=1e-10)
    assert sim.evaluate(a) == pytest.approx(value, abs=1e-10)
    np.testing.assert_allclose(sim.state(a), ref.state(a), atol=1e-12)
    for i in range(len(x)):
        d = np.zeros_like(x); d[i] = 1e-6
        fd = (ref.evaluate(Angles.unpack(x + d, p, cd)) - ref.evaluate(Angles.unpack(x - d, p, cd))) / 2e-6
        assert g[i] == pytest.approx(fd, abs=2e-7)


def test_lbfgs_uses_gradient_and_logs_trace():
    inst = make_instance(6, 'regular3', 'uniform', 12)
    sim = LightningTapeSimulator(inst, device='lightning.qubit')
    logs = []
    res = run_lbfgs(inst, 2, True, ramp_start(inst, 2, True), sim, maxiter=5, trace_callback=logs.append)
    assert res.evaluations == len(logs) and res.energy <= logs[0]['energy'] + 1e-12
    assert res.energy == pytest.approx(StatevectorSimulator(inst).evaluate(res.angles), abs=1e-10)
