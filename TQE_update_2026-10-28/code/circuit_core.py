"""PennyLane-only circuits, classical prediction, and durable result checks."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import csv,hashlib,json,math,tempfile
from pathlib import Path
import numpy as np
import pennylane as qml

KINDS=('u1','old','qaoa')
BACKEND='pennylane.lightning.qubit'
PROTOCOL='pl1050-v1-big-endian-cost-edge-YZ-ZY-mixer-c128'
ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT.parent

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def object_digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n'
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

def read_record(path,identity):
    try:r=json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,ValueError) as e:raise ValueError(f'Corrupt record: {path}') from e
    for k,v in identity.items():
        if r.get(k)!=v:raise ValueError(f'Record identity mismatch: {path}: {k}')
    if set(r.get('methods',{}))!=set(KINDS):raise ValueError(f'Missing methods: {path}')
    for k in KINDS:
        m=r['methods'][k]
        for key in ('probability','energy','ratio','norm_error'):
            if not math.isfinite(float(m.get(key,float('nan')))):raise ValueError(f'Invalid {key}: {path}')
        if not -1e-12<=m['probability']<=1+1e-12:raise ValueError(f'Invalid probability: {path}')
        if not -1e-9<=m['ratio']<=1+1e-9:raise ValueError(f'Invalid ratio: {path}')
        if abs(m['norm_error'])>1e-9:raise ValueError(f'Invalid normalization: {path}')
    return r

def fingerprint(g):
    return object_digest({k:g[k] for k in ('n','edges','weights')})

def scales(g):
    return 2*len(g['edges'])/g['n'],float(np.sqrt(np.mean(np.square(g['weights']))))

def predict(g,p,kind,models):
    D,s=scales(g);f=1. if D<=1 else math.atan(1/math.sqrt(D-1))
    if kind=='u1':
        m=models['u1'];t=m['templates'][str(p)];r=D/m['reference_degree']
        result=([x['gamma']['A']*r**x['gamma']['b']/s for x in t['layers']],t['beta'],
                [x['alpha']['A']*r**x['alpha']['b']/s for x in t['layers']])
    elif kind=='old':
        m=models['old'];t=m['templates'][str(p)]
        r=max(D,.1)/max(m['reference_degree'],.1)
        result=([x*f/s for x in t['gamma_unit']],t['beta'],
                [x/s*r**m['alpha_degree_power']*m['alpha_multiplier'] for x in t['alpha_unit']])
    elif kind=='qaoa':
        m=models['small' if g['n']<=16 else 'large'];method=m['qaoa_fixed_best_by_depth'][str(p)]
        if method=='qaoa_trained':
            t=m['templates'][str(p)]['qaoa']
            result=([x*f/s for x in t['gamma_unit']],t['beta'],[0.]*p)
        elif method in ('qaoa_csv','qaoa_csv_rms'):
            t=models['csv'][str(p)]
            result=([x*m.get('csv_sign',-1)*(f if method=='qaoa_csv' else 1)/s for x in t[:p]],t[p:],[0.]*p)
        else:raise ValueError(f'Unsupported frozen comparator {method}')
    else:raise ValueError(kind)
    result=tuple(np.array(x,dtype=float) for x in result)
    if any(len(x)!=p or not np.isfinite(x).all() for x in result):raise ValueError('Invalid angles')
    return result

def make_state_function(g,p,cd):
    dev=qml.device('lightning.qubit',wires=g['n'],c_dtype=np.complex128,shots=None)
    @qml.qnode(dev,interface=None,diff_method=None,cache=False)
    def state(gamma,beta,alpha):
        for i in range(g['n']):qml.Hadamard(wires=i)
        for l in range(p):
            for (i,j),w in zip(g['edges'],g['weights']):qml.IsingZZ(2*gamma[l]*w,wires=[i,j])
            if cd:
                for (i,j),w in zip(g['edges'],g['weights']):
                    qml.PauliRot(2*alpha[l]*w,'YZ',wires=[i,j])
                    qml.PauliRot(2*alpha[l]*w,'ZY',wires=[i,j])
            for i in range(g['n']):qml.RX(2*beta[l],wires=i)
        return qml.state()
    return state

def spectrum(g):
    n=g['n'];idx=np.arange(1<<n,dtype=np.uint32);v=np.zeros(1<<n)
    for (i,j),w in zip(g['edges'],g['weights']):
        parity=(((idx>>(n-1-i))^(idx>>(n-1-j)))&1).astype(np.int8)
        v+=w*(1-2*parity)
    return v

def optimal_mask(v,W):
    delta=np.abs(v-v.min());loose=delta<=1e-8*max(1.,W);tight=delta<=1e-10*max(1.,W)
    if not np.array_equal(loose,tight):raise ValueError('Ambiguous optimum mask: loose/tight tolerances differ; inspect energy gap')
    return tight

def measure(psi,v,mask):
    pr=np.abs(np.asarray(psi))**2;norm_error=float(abs(pr.sum()-1))
    if norm_error>1e-9:raise ValueError('State not normalized')
    E=float(np.dot(pr,v));ratio=float((v.max()-E)/(v.max()-v.min()))
    return {'probability':float(pr[mask].sum()),'energy':E,'ratio':ratio,'norm_error':norm_error}

def runtime():
    import platform,importlib.metadata
    return {'python':platform.python_version(),'numpy':np.__version__,'pennylane':qml.__version__,
            'pennylane_lightning':importlib.metadata.version('pennylane-lightning'),
            'backend':BACKEND,'dtype':'complex128','shots':None,'threads':4,'platform':platform.platform()}

