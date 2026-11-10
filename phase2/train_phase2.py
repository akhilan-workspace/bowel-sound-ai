"""Bowel-sound detection, phase 2: pretrained audio model (YAMNet) vs phase-1 baselines,
plus a recall-oriented post-processing study.  CPU-only, no internet.

Pretrained model: Google YAMNet (MobileNet-v1, AudioSet), attached as a Kaggle Model
(google/yamnet/tensorFlow2/yamnet).  YAMNet expects 16 kHz audio and emits one
1024-d embedding per 0.96 s patch with a fixed 0.48 s hop.  To get a dense
time grid we run it on 12 copies of each clip shifted by 40 ms, giving an embedding
every 40 ms, then interpolate to the 10 ms label frames.

Models compared (same leakage-controlled protocols as phase 1):
  yamnet_probe : frozen YAMNet embeddings -> small BiGRU head (pretrained-only)
  crnn         : phase-1 CRNN on log-mel (re-run here for a paired comparison)
  crnn_yamnet  : CRNN whose GRU also receives the YAMNet embeddings (fusion)
Post-processing (tuned on TRAINING validation data only):
  frame threshold that maximises EVENT F1 (instead of frame F1), median filter,
  minimum event length.  Reported alongside the plain frame-F1 threshold.

Protocols: official (authors' split) and grouped 5-fold by recording prefix.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, glob, json, time, math, argparse, random
import numpy as np, pandas as pd, soundfile as sf
from scipy.signal import resample_poly
from scipy.ndimage import median_filter
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

p = argparse.ArgumentParser()
p.add_argument('--data', default=None)
p.add_argument('--yamnet', default=None)
p.add_argument('--out', default='/kaggle/working/phase2')
p.add_argument('--epochs', type=int, default=30)
p.add_argument('--seeds', type=int, default=2)
p.add_argument('--folds', type=int, default=5)
p.add_argument('--models', default='yamnet_probe,crnn,crnn_yamnet')
p.add_argument('--quick', action='store_true')
A = p.parse_args()
os.makedirs(A.out, exist_ok=True)
T0 = time.time()
def log(*a):
    s = f"[{(time.time()-T0)/60:6.1f} min] " + ' '.join(str(x) for x in a)
    print(s, flush=True)
    with open(os.path.join(A.out, 'log.txt'), 'a') as f: f.write(s + '\n')

# ---------------------------------------------------------------- data
if A.data is None:
    hits = glob.glob('/kaggle/input/**/files.csv', recursive=True)
    if not hits: sys.exit('files.csv not found under /kaggle/input')
    A.data = os.path.dirname(hits[0])
if A.yamnet is None:
    hits = glob.glob('/kaggle/input/**/saved_model.pb', recursive=True)
    hits = [h for h in hits if 'yamnet' in h.lower()] or hits
    if not hits: sys.exit('YAMNet saved_model not found under /kaggle/input')
    A.yamnet = os.path.dirname(hits[0])
meta = pd.read_csv(os.path.join(A.data, 'files.csv'))
meta['stem'] = meta['filename'].str.replace('.wav', '', regex=False)
meta['rec'] = meta['stem'].str.split('_').str[0]
meta['split'] = meta['train/test'].str.strip().str.lower()
meta = meta[meta.stem.apply(lambda s: os.path.exists(os.path.join(A.data, s + '.wav')))].reset_index(drop=True)
if A.quick: meta = meta.sample(min(len(meta), 120), random_state=0).reset_index(drop=True)
log('clips', len(meta), 'recordings', meta.rec.nunique(), 'yamnet', A.yamnet)

HOP, NF, N_MELS, SR = 0.010, 200, 64, 8000
import librosa

def read(stem):
    y, sr = sf.read(os.path.join(A.data, stem + '.wav'), dtype='float32')
    if y.ndim > 1: y = y.mean(1)
    return y, sr

def rs(y, sr, to):
    g = math.gcd(to, sr); return resample_poly(y, to // g, sr // g).astype(np.float32)

def logmel(y, sr):
    y = rs(y, sr, SR); n = 2 * SR; y = np.pad(y, (0, max(0, n - len(y))))[:n]
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=256, hop_length=int(HOP * SR), win_length=200,
                                       n_mels=N_MELS, fmin=50, fmax=SR // 2, power=2.0)
    return np.log(m + 1e-8)[:, :NF].T.astype(np.float32)

def events(stem):
    f = os.path.join(A.data, stem + '.csv')
    if not os.path.exists(f) or os.path.getsize(f) == 0: return np.zeros((0, 2))
    e = pd.read_csv(f)
    if e.empty or 'start' not in e: return np.zeros((0, 2))
    return np.clip(e[['start', 'end']].astype(float).dropna().values, 0, 2.0)

def frame_labels(ev):
    lab = np.zeros(NF, np.float32); c = np.arange(NF) * HOP
    for s, e in ev: lab[(c + HOP / 2 > s) & (c - HOP / 2 < e)] = 1
    return lab

# ---------------------------------------------------------------- YAMNet dense embeddings
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count() or 2)
yam = tf.saved_model.load(A.yamnet)
YSR, PAD, SHIFT, NSH = 16000, 0.5, 0.04, 12           # 12 shifts x 40 ms = 0.48 s (= YAMNet hop)
FT = np.arange(NF) * HOP                              # label frame times

def yamnet_dense(y, sr):
    y = rs(y, sr, YSR); n = 2 * YSR; y = np.pad(y, (0, max(0, n - len(y))))[:n]
    y = y / (np.abs(y).max() + 1e-6) * 0.5                # YAMNet expects [-1, 1]
    y = np.pad(y, (int(PAD * YSR), int(PAD * YSR)), mode='reflect')
    T, E = [], []
    for k in range(NSH):
        off = int(k * SHIFT * YSR)
        _, emb, _ = yam(y[off:])
        emb = emb.numpy()
        centre = k * SHIFT + np.arange(len(emb)) * 0.48 + 0.48 - PAD
        T.append(centre); E.append(emb)
    T = np.concatenate(T); E = np.concatenate(E); o = np.argsort(T); T, E = T[o], E[o]
    idx = np.clip(np.searchsorted(T, FT), 1, len(T) - 1)  # linear interpolation to 10 ms frames
    w = ((FT - T[idx - 1]) / (T[idx] - T[idx - 1] + 1e-9)).clip(0, 1)[:, None]
    return ((1 - w) * E[idx - 1] + w * E[idx]).astype(np.float16)

cache = os.path.join(A.out, 'features.npz')
if os.path.exists(cache) and not A.quick:
    z = np.load(cache); X, Z = z['X'], z['Z'].astype(np.float32); log('loaded cached features')
else:
    log('extracting log-mel + dense YAMNet embeddings ...')
    X, Z = [], []
    for i, s in enumerate(meta.stem):
        y, sr = read(s); X.append(logmel(y, sr)); Z.append(yamnet_dense(y, sr))
        if i % 200 == 0: log('  features', i, '/', len(meta))
    X = np.stack(X); Z = np.stack(Z)
    if not A.quick: np.savez_compressed(cache, X=X, Z=Z)
    Z = Z.astype(np.float32)
del yam
EV = [events(s) for s in meta.stem]
Y = np.stack([frame_labels(e) for e in EV])
log('X', X.shape, 'Z', Z.shape, 'pos rate %.3f' % Y.mean())

# ---------------------------------------------------------------- metrics + post-processing
def to_events(b, gap=2):
    ev, on = [], None
    for i, v in enumerate(list(b) + [0]):
        if v and on is None: on = i
        if not v and on is not None: ev.append([on, i]); on = None
    merged = []
    for e in ev:
        if merged and e[0] - merged[-1][1] <= gap: merged[-1][1] = e[1]
        else: merged.append(e)
    return [(a * HOP, b_ * HOP) for a, b_ in merged]

def event_prf(pred_bin, ev_list):
    tp_p = n_p = tp_g = n_g = 0
    for pb, gt in zip(pred_bin, ev_list):
        pe = to_events(pb); n_p += len(pe); n_g += len(gt); used = set()
        for gs, ge in gt:
            for j, (ps, pe_) in enumerate(pe):
                if j not in used and ps < ge and pe_ > gs: used.add(j); tp_g += 1; break
        tp_p += len(used)
    P = tp_p / n_p if n_p else 0.0; R = tp_g / n_g if n_g else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)

def postproc(prob, thr, med, minlen):
    pr = median_filter(prob, size=(1, med)) if med > 1 else prob
    b = (pr > thr).astype(int)
    if minlen > 1:
        for r in b:
            on = None
            for i, v in enumerate(list(r) + [0]):
                if v and on is None: on = i
                if not v and on is not None:
                    if i - on < minlen: r[on:i] = 0
                    on = None
    return b

def tune_frame(prob, y):
    best = (0.5, -1)
    for t in np.linspace(0.05, 0.95, 37):
        f = f1_score(y.ravel(), (prob.ravel() > t).astype(int), zero_division=0)
        if f > best[1]: best = (t, f)
    return dict(thr=best[0], med=1, minlen=1)

def tune_event(prob, ev_list):
    best = (None, -1)
    for med in (1, 3, 5):
        for minlen in (1, 2, 3):
            for t in np.linspace(0.05, 0.95, 19):
                F = event_prf(postproc(prob, t, med, minlen), ev_list)[2]
                if F > best[1]: best = (dict(thr=t, med=med, minlen=minlen), F)
    return best[0]

def score(prob, y, ev_list, pp):
    b = postproc(prob, pp['thr'], pp['med'], pp['minlen'])
    yt, pt = y.ravel(), prob.ravel()
    P, R, F = event_prf(b, ev_list)
    return {'frame_auroc': float(roc_auc_score(yt, pt)) if 0 < yt.mean() < 1 else float('nan'),
            'frame_ap': float(average_precision_score(yt, pt)),
            'frame_f1': float(f1_score(yt, b.ravel(), zero_division=0)),
            'event_precision': P, 'event_recall': R, 'event_f1': F,
            'threshold': float(pp['thr']), 'median': pp['med'], 'min_len': pp['minlen']}

# ---------------------------------------------------------------- models
import torch, torch.nn as nn
torch.set_num_threads(max(1, os.cpu_count() or 1))

class CRNN(nn.Module):
    def __init__(s, nm=N_MELS, ch=64, hid=96, zdim=0):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                            nn.MaxPool2d((1, 2)), nn.Dropout(0.2))
        s.cnn = nn.Sequential(blk(1, ch // 2), blk(ch // 2, ch), blk(ch, ch), blk(ch, ch))
        s.zp = nn.Sequential(nn.Linear(zdim, 64), nn.ReLU(), nn.Dropout(0.3)) if zdim else None
        s.rnn = nn.GRU(ch * (nm // 16) + (64 if zdim else 0), hid, batch_first=True, bidirectional=True)
        s.out = nn.Linear(2 * hid, 1)
    def forward(s, x, z=None):
        h = s.cnn(x.unsqueeze(1)).permute(0, 2, 1, 3).flatten(2)
        if s.zp is not None: h = torch.cat([h, s.zp(z)], -1)
        h, _ = s.rnn(h); return s.out(h).squeeze(-1)

class Probe(nn.Module):                                   # pretrained embeddings only
    def __init__(s, zdim=1024, hid=96):
        super().__init__()
        s.zp = nn.Sequential(nn.Linear(zdim, 128), nn.ReLU(), nn.Dropout(0.3))
        s.rnn = nn.GRU(128, hid, batch_first=True, bidirectional=True); s.out = nn.Linear(2 * hid, 1)
    def forward(s, x, z):
        h, _ = s.rnn(s.zp(z)); return s.out(h).squeeze(-1)

def fit(kind, tr, va, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    mu, sd = X[tr].mean((0, 1)), X[tr].std((0, 1)) + 1e-5
    zmu, zsd = Z[tr].mean((0, 1)), Z[tr].std((0, 1)) + 1e-5
    nx = lambda i: torch.tensor((X[i] - mu) / sd); nzz = lambda i: torch.tensor((Z[i] - zmu) / zsd)
    net = {'crnn': lambda: CRNN(), 'crnn_yamnet': lambda: CRNN(zdim=Z.shape[2]), 'yamnet_probe': lambda: Probe(Z.shape[2])}[kind]()
    E = A.epochs if not A.quick else 2
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=E * math.ceil(len(tr) / 32))
    pw = torch.tensor((1 - Y[tr].mean()) / max(Y[tr].mean(), 1e-3)).clamp(max=10)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    xt, zt, yt = nx(tr), nzz(tr), torch.tensor(Y[tr]); xv, zv = nx(va), nzz(va)
    best, state = -1, None
    for ep in range(E):
        net.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 32):
            b = perm[i:i + 32]; xb = xt[b].clone(); zb = zt[b]
            for j in range(len(xb)):
                f0 = np.random.randint(0, N_MELS - 8); xb[j, :, f0:f0 + np.random.randint(0, 8)] = 0
            xb = xb + 0.1 * torch.randn(len(xb), 1, 1)
            loss = lossf(net(xb, zb), yt[b])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5); opt.step(); sch.step()
        net.eval()
        with torch.no_grad(): pv = torch.sigmoid(net(xv, zv)).numpy()
        ap = average_precision_score(Y[va].ravel(), pv.ravel()) if Y[va].sum() else 0
        if ap > best: best, state = ap, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state); net.eval()
    def pred(i):
        with torch.no_grad(): return torch.sigmoid(net(nx(i), nzz(i))).numpy()
    return pred

# ---------------------------------------------------------------- run
allrows = []
def run_split(name, tr_idx, te_idx, seed):
    a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(tr_idx, groups=meta.rec.values[tr_idx]))
    fit_i, val_i = tr_idx[a], tr_idx[b]
    for m in A.models.split(','):
        t = time.time(); f = fit(m, fit_i, val_i, seed)
        pv, pt = f(val_i), f(te_idx); EVv = [EV[i] for i in val_i]; EVt = [EV[i] for i in te_idx]
        for pp_name, pp in (('frameF1_thr', tune_frame(pv, Y[val_i])), ('eventF1_tuned', tune_event(pv, EVv))):
            r = score(pt, Y[te_idx], EVt, pp)
            r.update(model=m, postproc=pp_name, protocol=name, seed=seed, n_train=len(fit_i), n_test=len(te_idx),
                     fit_minutes=round((time.time() - t) / 60, 2))
            allrows.append(r)
            log(name, m, pp_name, 'seed', seed, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})
        pd.DataFrame(allrows).to_csv(os.path.join(A.out, 'results_all.csv'), index=False)

idx = np.arange(len(meta))
for seed in range(A.seeds if not A.quick else 1):
    run_split('official', idx[meta.split == 'train'], idx[meta.split == 'test'], seed)
    for k, (a, b) in enumerate(GroupKFold(A.folds if not A.quick else 2).split(idx, groups=meta.rec.values)):
        run_split(f'grouped_fold{k}', a, b, seed)

df = pd.DataFrame(allrows)
df['protocol_group'] = df.protocol.str.replace(r'_fold\d+', '', regex=True)
mets = ['frame_auroc', 'frame_ap', 'frame_f1', 'event_precision', 'event_recall', 'event_f1']
summ = df.groupby(['protocol_group', 'model', 'postproc'])[mets].agg(['mean', 'std']).round(4)
summ.to_csv(os.path.join(A.out, 'results_summary.csv'))
log('\n' + summ.to_string())
log('DONE')
