import json
from pathlib import Path
import numpy as np
from cdtransfer.instances import make_splits,make_instance
from cdtransfer.simulator import StatevectorSimulator
from cdtransfer.transfer import predict
from cdtransfer.experiment import bootstrap_ci


def test_disjoint_splits():
    path=Path(__file__).resolve().parents[1]/"configs/pilot.json"
    conf=json.loads(path.read_text())
    splits=make_splits(conf)
    groups=[{r["fingerprint"] for r in splits[x]} for x in ("train","val","test")]
    assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    assert len(splits["test"])==4


def test_predict_stays_pure(monkeypatch):
    inst=make_instance(6,"ring","uniform",123)
    def forbidden(*args,**kwargs):
        raise AssertionError("Simulator/energy must not be called by fixed-angle predictor")
    monkeypatch.setattr(StatevectorSimulator,"evaluate",forbidden)
    monkeypatch.setattr(StatevectorSimulator,"state",forbidden)
    a=predict(inst,1,"qaoa_csv")
    b=predict(inst,1,"cd_bch")
    assert np.isfinite(a.gamma).all() and np.isfinite(b.alpha).all()


def test_bootstrap_ci():
    lo,hi=bootstrap_ci([1.,2.,3.],n_boot=100)
    assert lo<=2<=hi
