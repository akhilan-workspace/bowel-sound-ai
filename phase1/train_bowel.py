"""Bowel-sound detection: leakage-controlled comparison (phase 1: classical + CRNN).

Data: Kaggle robertnowak/bowel-sounds (Ficek et al. 2021, CC BY-NC 4.0).
Each N_x.wav is a 2 s clip; N_x.csv lists bowel-sound events (start, end in s).
files.csv gives the authors' official train/test split.

Task: frame-level detection on 10 ms frames, plus event-level F1.

Evaluation protocols
  official : authors' train/test split (comparability with the paper)
  grouped  : 5-fold GroupKFold by recording prefix (N in N_x) so neighbouring
             clips of one recording never sit in both train and test.
Decision thresholds are tuned on an inner validation split of the TRAINING
data only, never on test.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, glob, json, time, math, argparse, random
import numpy as np, pandas as pd, soundfile as sf
from scipy.signal import resample_poly
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

p = argparse.ArgumentParser()
p.add_argument('--data', default=None)
p.add_argument('--out', default='/kaggle/working/results')
p.add_argument('--sr', type=int, default=8000)
p.add_argument('--epochs', type=int, default=30)
p.add_argument('--seeds', type=int, default=2)
p.add_argument('--folds', type=int, default=5)
p.add_argument('--models', default='lgbm,crnn')
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
meta = pd.read_csv(os.path.join(A.data, 'files.csv'))
meta['stem'] = meta['filename'].str.replace('.wav', '', regex=False)
meta['rec'] = meta['stem'].str.split('_').str[0]
meta['split'] = meta['train/test'].str.strip().str.lower()
meta = meta[meta.stem.apply(lambda s: os.path.exists(os.path.join(A.data, s + '.wav')))].reset_index(drop=True)
if A.quick: meta = meta.sample(min(len(meta), 200), random_state=0).reset_index(drop=True)
log('clips', len(meta), 'recordings', meta.rec.nunique(), 'official split', meta.split.value_counts().to_dict())

HOP = 0.010                       # 10 ms frames, as in the original paper
NF = 200                          # frames per 2 s clip
N_MELS = 64
import librosa

def load_clip(stem):
    y, sr = sf.read(os.path.join(A.data, stem + '.wav'), dtype='float32')
    if y.ndim > 1: y = y.mean(1)
    g = math.gcd(A.sr, sr); y = resample_poly(y, A.sr // g, sr // g).astype(np.float32)
    n = int(2.0 * A.sr); y = np.pad(y, (0, max(0, n - len(y))))[:n]
    hop = int(HOP * A.sr)
    m = librosa.feature.melspectrogram(y=y, sr=A.sr, n_fft=256, hop_length=hop, win_length=200,
                                       n_mels=N_MELS, fmin=50, fmax=A.sr // 2, power=2.0)
    return np.log(m + 1e-8)[:, :NF].T.astype(np.float32)        # (NF, N_MELS)

def events(stem):
    f = os.path.join(A.data, stem + '.csv')
    if not os.path.exists(f) or os.path.getsize(f) == 0: return np.zeros((0, 2))
    e = pd.read_csv(f)
    if e.empty or 'start' not in e: return np.zeros((0, 2))
    e = e[['start', 'end']].astype(float).dropna().values
    return np.clip(e, 0, 2.0)

def frame_labels(ev):
    lab = np.zeros(NF, np.float32)
    c = np.arange(NF) * HOP                                     # frame centres (librosa center=True)
    for s, e in ev:
        lab[(c + HOP / 2 > s) & (c - HOP / 2 < e)] = 1
    return lab

log('extracting features ...')
X = np.stack([load_clip(s) for s in meta.stem])
EV = [events(s) for s in meta.stem]
Y = np.stack([frame_labels(e) for e in EV])
log('X', X.shape, 'positive frame rate %.3f' % Y.mean(), 'events', sum(len(e) for e in EV))

# leakage check of the official split
tr_rec = set(meta.rec[meta.split == 'train']); te_rec = set(meta.rec[meta.split == 'test'])
leak = {'test_recordings': len(te_rec), 'test_recordings_also_in_train': len(te_rec & tr_rec),
        'test_clips_whose_recording_is_in_train': int(meta[(meta.split == 'test') & meta.rec.isin(tr_rec)].shape[0]),
        'test_clips': int((meta.split == 'test').sum())}
log('official-split leakage check', leak)
json.dump(leak, open(os.path.join(A.out, 'official_split_leakage.json'), 'w'), indent=1)

# ---------------------------------------------------------------- metrics
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
        pe = to_events(pb); n_p += len(pe); n_g += len(gt)
        used = set()
        for gs, ge in gt:
            for j, (ps, pe_) in enumerate(pe):
                if j not in used and ps < ge and pe_ > gs: used.add(j); tp_g += 1; break
        tp_p += len(used)
    P = tp_p / n_p if n_p else 0.0; R = tp_g / n_g if n_g else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)

def tune_threshold(prob, y):
    best = (0.5, -1)
    for t in np.linspace(0.05, 0.95, 37):
        f = f1_score(y.ravel(), (prob.ravel() > t).astype(int), zero_division=0)
        if f > best[1]: best = (t, f)
    return best[0]

def score(prob, y, ev_list, thr):
    b = (prob > thr).astype(int)
    yt, pt = y.ravel(), prob.ravel()
    P, R, F = event_prf(b, ev_list)
    return {'frame_auroc': float(roc_auc_score(yt, pt)) if 0 < yt.mean() < 1 else float('nan'),
            'frame_ap': float(average_precision_score(yt, pt)),
            'frame_f1': f1_score(yt, b.ravel(), zero_division=0),
            'frame_acc': float((b.ravel() == yt).mean()),
            'event_precision': P, 'event_recall': R, 'event_f1': F, 'threshold': float(thr)}

# ---------------------------------------------------------------- models
def ctx_features(x):                    # x: (n, NF, N_MELS) -> (n*NF, d)
    d = np.diff(x, axis=1, prepend=x[:, :1])
    k = 5; pad = np.pad(x, ((0, 0), (k, k), (0, 0)), mode='edge')
    win = np.stack([pad[:, i:i + NF] for i in range(2 * k + 1)], 0)
    feats = [x, d, win.mean(0), win.std(0), x - win.mean(0)]
    e = x.mean(2, keepdims=True); feats.append(e)
    return np.concatenate(feats, 2).reshape(-1, feats[0].shape[2] * 5 + 1)

def fit_lgbm(Xtr, Ytr, Xva, seed):
    import lightgbm as lgb
    pos = Ytr.mean()
    m = lgb.LGBMClassifier(n_estimators=600 if not A.quick else 50, learning_rate=0.05, num_leaves=63,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                           scale_pos_weight=(1 - pos) / max(pos, 1e-3), random_state=seed, verbose=-1)
    m.fit(ctx_features(Xtr), Ytr.ravel())
    return lambda Z: m.predict_proba(ctx_features(Z))[:, 1].reshape(len(Z), NF)

import torch, torch.nn as nn
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(max(1, os.cpu_count() or 1))

class CRNN(nn.Module):
    def __init__(s, nm=N_MELS, ch=64, hid=96):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                            nn.MaxPool2d((1, 2)), nn.Dropout(0.2))
        s.cnn = nn.Sequential(blk(1, ch // 2), blk(ch // 2, ch), blk(ch, ch), blk(ch, ch))
        s.rnn = nn.GRU(ch * (nm // 16), hid, batch_first=True, bidirectional=True)
        s.out = nn.Linear(2 * hid, 1)
    def forward(s, x):                                  # x: (b, NF, nm)
        h = s.cnn(x.unsqueeze(1))                       # (b, ch, NF, nm/16)
        h = h.permute(0, 2, 1, 3).flatten(2)
        h, _ = s.rnn(h)
        return s.out(h).squeeze(-1)

def fit_crnn(Xtr, Ytr, Xva, Yva, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    mu, sd = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-5
    nz = lambda Z: torch.tensor((Z - mu) / sd)
    net = CRNN().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    E = A.epochs if not A.quick else 3
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=E * math.ceil(len(Xtr) / 32))
    pw = torch.tensor((1 - Ytr.mean()) / max(Ytr.mean(), 1e-3)).clamp(max=10).to(DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    xt, yt, xv = nz(Xtr), torch.tensor(Ytr), nz(Xva).to(DEV)
    best, best_state = -1, None
    for ep in range(E):
        net.train(); perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 32):
            idx = perm[i:i + 32]; xb = xt[idx].clone()
            # SpecAugment-lite: random frequency mask + random gain
            for b in range(len(xb)):
                f0 = np.random.randint(0, N_MELS - 8); xb[b, :, f0:f0 + np.random.randint(0, 8)] = 0
            xb = xb + 0.1 * torch.randn(len(xb), 1, 1)
            loss = lossf(net(xb.to(DEV)), yt[idx].to(DEV))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5); opt.step(); sch.step()
        net.eval()
        with torch.no_grad(): pv = torch.sigmoid(net(xv)).cpu().numpy()
        ap = average_precision_score(Yva.ravel(), pv.ravel()) if Yva.sum() else 0
        if ap > best: best, best_state = ap, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    def pred(Z):
        with torch.no_grad(): return torch.sigmoid(net(nz(Z).to(DEV))).cpu().numpy()
    pred.net = net
    return pred

# ---------------------------------------------------------------- run
def run_split(name, tr_idx, te_idx, seed):
    rows = []
    gss = GroupShuffleSplit(1, test_size=0.15, random_state=seed)
    a, b = next(gss.split(tr_idx, groups=meta.rec.values[tr_idx]))
    fit_i, val_i = tr_idx[a], tr_idx[b]
    for mname in A.models.split(','):
        t = time.time()
        if mname == 'lgbm': f = fit_lgbm(X[fit_i], Y[fit_i], X[val_i], seed)
        elif mname == 'crnn': f = fit_crnn(X[fit_i], Y[fit_i], X[val_i], Y[val_i], seed)
        else: continue
        thr = tune_threshold(f(X[val_i]), Y[val_i])
        pt = f(X[te_idx])
        r = score(pt, Y[te_idx], [EV[i] for i in te_idx], thr)
        r.update(model=mname, protocol=name, seed=seed, n_train=len(fit_i), n_test=len(te_idx),
                 fit_minutes=round((time.time() - t) / 60, 2))
        allrows.append(r); pd.DataFrame(allrows).to_csv(os.path.join(A.out, 'results_all.csv'), index=False)
        rows.append(r); log(name, mname, 'seed', seed, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})
        if mname == 'crnn' and name.startswith('official') and seed == 0:
            torch.save(f.net.state_dict(), os.path.join(A.out, 'crnn_official_seed0.pt'))
            np.save(os.path.join(A.out, 'crnn_official_test_probs.npy'), pt)
    return rows

allrows = []
idx = np.arange(len(meta))
for seed in range(A.seeds if not A.quick else 1):
    run_split('official', idx[meta.split == 'train'], idx[meta.split == 'test'], seed)
    gkf = GroupKFold(A.folds)
    for k, (a, b) in enumerate(gkf.split(idx, groups=meta.rec.values)):
        run_split(f'grouped_fold{k}', a, b, seed)

df = pd.DataFrame(allrows)
df['protocol_group'] = df.protocol.str.replace(r'_fold\d+', '', regex=True)
metrics = ['frame_auroc', 'frame_ap', 'frame_f1', 'frame_acc', 'event_precision', 'event_recall', 'event_f1']
summ = df.groupby(['protocol_group', 'model'])[metrics].agg(['mean', 'std']).round(4)
summ.to_csv(os.path.join(A.out, 'results_summary.csv'))
log('\n' + summ.to_string())

# ---------------------------------------------------------------- figures
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, m in enumerate(['frame_f1', 'event_f1']):
        g = df.groupby(['model', 'protocol_group'])[m].agg(['mean', 'std']).unstack()
        g['mean'].plot.bar(yerr=g['std'], ax=ax[i], capsize=3, rot=0); ax[i].set_title(m); ax[i].set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(os.path.join(A.out, 'protocol_comparison.png'), dpi=150)
    if os.path.exists(os.path.join(A.out, 'crnn_official_test_probs.npy')):
        pt = np.load(os.path.join(A.out, 'crnn_official_test_probs.npy'))
        te = idx[meta.split == 'test']
        fig, ax = plt.subplots(4, 1, figsize=(10, 9))
        for j, i in enumerate(te[:4]):
            ax[j].imshow(X[i].T, aspect='auto', origin='lower', extent=[0, 2, 0, N_MELS], cmap='magma')
            t = np.arange(NF) * HOP
            ax[j].plot(t, Y[i] * N_MELS * 0.9, 'c', lw=1, label='expert')
            ax[j].plot(t, pt[j] * N_MELS * 0.9, 'w', lw=1, label='CRNN prob')
            ax[j].set_title(meta.stem[i]); ax[j].legend(loc='upper right', fontsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(A.out, 'crnn_examples.png'), dpi=120)
except Exception as e:
    log('plotting failed', e)
log('DONE')
