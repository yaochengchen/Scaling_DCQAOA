from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from cdtransfer.instances import IsingInstance
from cdtransfer.simulator import Angles, StatevectorSimulator
from cdtransfer.transfer import qaoa_csv, load_source_angles, predict
from cdtransfer.resources import comparable_qaoa_depth, gate_cost

X = np.array([[0,1],[1,0]], complex)
Y = np.array([[0,-1j],[1j,0]], complex)
Z = np.diag([1,-1]).astype(complex)
I = np.eye(2, dtype=complex)


def inst2(w=1.3):
    return IsingInstance(2,np.array([[0,1]]),np.array([w]))


def test_csv_correct_parser():
    a=load_source_angles(3)
    np.testing.assert_allclose(a.gamma,[0.3297,0.5688,0.6406])
    np.testing.assert_allclose(a.beta,[0.55,0.3675,0.2109])


def test_cd_zero_is_standard_qaoa():
    inst=inst2()
    sim=StatevectorSimulator(inst)
    angles=Angles([0.3,0.7],[0.4,0.1],[0.0,0.0])
    ref=Angles(angles.gamma,angles.beta,[0,0])
    np.testing.assert_allclose(sim.state(angles),sim.state(ref),atol=1e-14)
    assert sim.evaluate(angles)==sim.evaluate(ref)


def test_single_layer_against_dense_expm():
    w=1.3; gamma=0.23; beta=-0.17; alpha=0.11
    sim=StatevectorSimulator(inst2(w))
    got=sim.state(Angles([gamma],[beta],[alpha]))
    hc=w*np.kron(Z,Z)
    hb=np.kron(X,I)+np.kron(I,X)
    yz=w*np.kron(Z,Y)  # Y on LSB qubit 0
    zy=w*np.kron(Y,Z)  # Y on MSB qubit 1
    plus=np.ones(4,dtype=complex)/2
    expected=expm(-1j*beta*hb)@expm(-1j*alpha*zy)@expm(-1j*alpha*yz)@expm(-1j*gamma*hc)@plus
    np.testing.assert_allclose(got,expected,atol=1e-12)
    comm=0.5j*(hb@hc-hc@hb)
    np.testing.assert_allclose(comm,yz+zy,atol=1e-13)


def test_scale_covariance_and_transfer_no_online():
    inst=inst2(0.7)
    sim=StatevectorSimulator(inst)
    a=qaoa_csv(inst,2)
    s=3.1
    a2=qaoa_csv(inst.scaled(s),2)
    np.testing.assert_allclose(a2.gamma,a.gamma/s)
    np.testing.assert_allclose(a2.beta,a.beta)
    cd=Angles(a.gamma,a.beta,np.array([0.3,0.15]))
    cd_scaled=Angles(cd.gamma/s,cd.beta,cd.alpha/s)
    np.testing.assert_allclose(sim.state(cd),StatevectorSimulator(inst.scaled(s)).state(cd_scaled),atol=1e-12)
    # predictor returns angles from static graph features, without evaluator
    pred=predict(inst,2,"cd_bch",{"csv_sign":-1,"bch_c":0.2})
    assert len(pred.alpha)==2


def test_gate_budget_equal_unoptimized():
    inst=inst2()
    cd=Angles([0.4,0.3],[0.1,0.2],[0.1,0.1])
    q=Angles([0.1]*6,[0.2]*6,[0]*6)
    assert comparable_qaoa_depth(cd.p)==q.p
    assert gate_cost(inst,cd)["decomposed_cx"]==gate_cost(inst,q)["decomposed_cx"]
    assert gate_cost(inst,cd)["decomposed_cx"]==12
