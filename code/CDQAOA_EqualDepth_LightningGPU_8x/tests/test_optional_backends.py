import numpy as np
import pytest
from cdtransfer.instances import IsingInstance
from cdtransfer.simulator import Angles, StatevectorSimulator


def test_qulacs_state_if_available():
    pytest.importorskip("qulacs")
    from cdtransfer.qulacs_optional import state
    inst=IsingInstance(3,[[0,1],[1,2]],[1.2,-0.7])
    a=Angles([0.21,0.45],[-0.2,0.1],[0.19,-0.15])
    np.testing.assert_allclose(state(inst,a),StatevectorSimulator(inst).state(a),atol=1e-10)


def test_qiskit_state_if_available():
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector
    from cdtransfer.qiskit_optional import circuit
    inst=IsingInstance(3,[[0,1],[1,2]],[1.2,-0.7])
    a=Angles([0.21,0.45],[-0.2,0.1],[0.19,-0.15])
    np.testing.assert_allclose(Statevector.from_instruction(circuit(inst,a)).data,
                               StatevectorSimulator(inst).state(a),atol=1e-10)
