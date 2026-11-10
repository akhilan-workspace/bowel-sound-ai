"""Bowel-sound detection, phase 3 (GPU):
leakage-controlled benchmark of classical, from-scratch deep, and pretrained audio models,
with cross-device external validation and few-subject adaptation.

Internal data : Kaggle robertnowak/bowel-sounds (Ficek et al., Sensors 2021; CC BY-NC 4.0)
                1606 x 2 s clips, 44.1 kHz, contact microphone, Poznan/Warsaw. 6413 annotated events.
External data : figshare 10.6084/m9.figshare.28595741 (Oldenburg; CC BY 4.0), 7 recordings from
                4 subjects, SonicGuard wearable sensor, 48 kHz, events typed SB/MB/CRS/HS.
                NEVER used for training or threshold selection in the external test.

Models
  lgbm   : LightGBM on per-frame log-mel + context features (phase 1)
  crnn   : CNN + BiGRU on log-mel, trained from scratch (phase 1)
  cnn14  : PANNs CNN14 conv blocks 1-4 pretrained on AudioSet (Kong et al. 2020), pooling changed to
           frequency-only so the 10 ms time resolution is kept, + BiGRU head, fine-tuned
  hubert : HuBERT-base (self-supervised speech model, 20 ms frames), fine-tuned, upsampled to 10 ms
Protocols (internal)
  random  : 5-fold KFold over clips (ignores recording identity; leaky by construction)
  grouped : 5-fold GroupKFold by recording prefix (leak-free) - PRIMARY
  official: authors' train/test split (for comparability with the literature)
External
  ext     : ensemble of the 5 grouped-fold models (mean probability), median fold threshold, applied to
            Oldenburg with 2 s windows / 1 s hop. Event metrics only (annotation conventions differ).
Adaptation (leave-one-subject-out on Oldenburg, 4 subjects)
  warsaw+old : train on all Warsaw + 3 Oldenburg subjects, test on the 4th
  old_only   : train on the 3 Oldenburg subjects only
Statistics
  pooled out-of-fold predictions, recording-level bootstrap 95% CIs (1000 resamples),
  paired bootstrap differences between models, leakage inflation (random - grouped).
All thresholds are tuned on an inner validation split of the training data only.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, glob, json, time, math, argparse, random, subprocess, copy
import numpy as np, pandas as pd, soundfile as sf
from scipy.signal import resample_poly
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, ShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

p = argparse.ArgumentParser()
p.add_argument('--data', default=None)
p.add_argument('--old', default='/kaggle/working/old')
p.add_argument('--cache', default='/tmp/cache')
p.add_argument('--out', default='/kaggle/working/phase3')
p.add_argument('--models', default='lgbm,crnn,cnn14,hubert')
p.add_argument('--seeds', type=int, default=3)
p.add_argument('--folds', type=int, default=5)
p.add_argument('--loso_models', default='crnn,cnn14')
p.add_argument('--loso_seeds', type=int, default=2)
p.add_argument('--ep_crnn', type=int, default=40)
p.add_argument('--ep_cnn14', type=int, default=25)
p.add_argument('--ep_hubert', type=int, default=15)
p.add_argument('--hubert_name', default='facebook/hubert-base-ls960')
p.add_argument('--cnn14_url', default='https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1')
p.add_argument('--analyze_only', action='store_true')
p.add_argument('--quick', action='store_true')
p.add_argument('--no_old', action='store_true')
A = p.parse_args()
for d in (A.out, A.cache, A.old): os.makedirs(d, exist_ok=True)
T0 = time.time()
def log(*a):
    s = f"[{(time.time()-T0)/60:7.1f} min] " + ' '.join(str(x) for x in a)
    print(s, flush=True)
    with open(os.path.join(A.out, 'log.txt'), 'a') as f: f.write(s + '\n')

HOP, NF, N_MELS = 0.010, 200, 64
MODELS = [m for m in A.models.split(',') if m]
SEEDS = list(range(1 if A.quick else A.seeds))
EP = dict(crnn=A.ep_crnn, cnn14=A.ep_cnn14, hubert=A.ep_hubert)
if A.quick: EP = dict(crnn=2, cnn14=2, hubert=1)

# ======================================================================== audio utils
def rs(y, sr, to):
    if sr == to: return y.astype(np.float32)
    g = math.gcd(to, sr); return resample_poly(y, to // g, sr // g).astype(np.float32)

def nf_gain(y16):
    """noise-floor normalisation: scale so the median 20 ms frame RMS equals 1e-3 (~ -60 dBFS).
    Makes recordings from different devices/gains comparable without squashing events."""
    n = 320; m = len(y16) // n
    if m < 3: return 1.0
    pw = (y16[:m * n].reshape(m, n) ** 2).mean(1)
    fl = math.sqrt(max(np.median(pw), 1e-20))
    return float(np.clip(1e-3 / fl, 1e-2, 1e5))

import librosa
MEL8 = librosa.filters.mel(sr=8000, n_fft=256, n_mels=N_MELS, fmin=50, fmax=4000).astype(np.float32)
def logmel8(y8):
    S = np.abs(librosa.stft(y8, n_fft=256, hop_length=80, win_length=200, center=True)) ** 2
    return np.log(MEL8 @ S + 1e-8)[:, :NF].T.astype(np.float32)

def fix(y, n): return np.pad(y, (0, max(0, n - len(y))))[:n]

def make_bank(waves, srs, labels):
    """waves: list of 2 s waveforms at original sr. Returns dict of model inputs."""
    X8, W16, W32 = [], [], []
    for y, sr in zip(waves, srs):
        y16 = fix(rs(y, sr, 16000), 32000); g = nf_gain(y16)
        y8 = fix(rs(y, sr, 8000), 16000) * g; y32 = fix(rs(y, sr, 32000), 64000) * g
        X8.append(logmel8(y8)); W16.append(y16 * g); W32.append(y32)
    return dict(X8=np.stack(X8), W16=np.stack(W16).astype(np.float32), W32=np.stack(W32).astype(np.float32),
                Y=np.stack(labels).astype(np.float32))

def cat_banks(a, ia, b, ib):
    return {k: np.concatenate([a[k][ia], b[k][ib]]) for k in a}

# ======================================================================== Warsaw data
if A.data is None:
    hits = glob.glob('/kaggle/input/**/files.csv', recursive=True)
    if not hits: sys.exit('files.csv not found')
    A.data = os.path.dirname(hits[0])
meta = pd.read_csv(os.path.join(A.data, 'files.csv'))
meta['stem'] = meta['filename'].str.replace('.wav', '', regex=False)
meta['rec'] = meta['stem'].str.split('_').str[0]
meta['split'] = meta['train/test'].str.strip().str.lower()
meta = meta[meta.stem.apply(lambda s: os.path.exists(os.path.join(A.data, s + '.wav')))].reset_index(drop=True)
if A.quick: meta = meta[meta.rec.isin(meta.rec.drop_duplicates().sample(min(60, meta.rec.nunique()), random_state=0))].reset_index(drop=True)

def w_events(stem):
    f = os.path.join(A.data, stem + '.csv')
    if not os.path.exists(f) or os.path.getsize(f) == 0: return pd.DataFrame(columns=['start', 'end', 'category'])
    e = pd.read_csv(f)
    if e.empty or 'start' not in e: return pd.DataFrame(columns=['start', 'end', 'category'])
    e = e.dropna(subset=['start', 'end']).copy()
    e['start'] = e.start.astype(float).clip(0, 2); e['end'] = e.end.astype(float).clip(0, 2)
    if 'category' not in e: e['category'] = np.nan
    return e[['start', 'end', 'category']]

def frame_labels(iv, n=NF):
    lab = np.zeros(n, np.float32); c = np.arange(n) * HOP
    for s, e in iv: lab[(c + HOP / 2 > s) & (c - HOP / 2 < e)] = 1
    return lab

WEV = [w_events(s) for s in meta.stem]
EVL = [ev[['start', 'end']].values for ev in WEV]
log('Warsaw clips', len(meta), 'recordings', meta.rec.nunique(), 'events', sum(len(e) for e in EVL))

# ======================================================================== Oldenburg data
OLD_FILES = {'record_300424001_1': ('52988447', '52988432'), 'record_300424001_2': ('52988450', '52988435'),
             'record_080524001_1': ('52988453', '52988438'), 'record_080524002_2': ('52988441', '52988420'),
             'record_100524001_1': ('52988456', '52988423'), 'record_100524001_2': ('52988459', '52988426'),
             'record_150524003_1': ('52988444', '52988429')}
def subj(name): return 'd' + name.split('_')[1][:6]         # recording date; 080524001/002 grouped conservatively
OLD = {}
if not A.no_old:
    for n, (w, t) in OLD_FILES.items():
        for fid, ext in ((w, '.wav'), (t, '.txt')):
            fp = os.path.join(A.old, n + ext)
            if not os.path.exists(fp) or os.path.getsize(fp) < 100:
                subprocess.run(['curl', '-sL', '--retry', '3', '-o', fp, f'https://ndownloader.figshare.com/files/{fid}'])
        y, sr = sf.read(os.path.join(A.old, n + '.wav'), dtype='float32')
        if y.ndim > 1: y = y.mean(1)
        if A.quick: y = y[:sr * 120]
        a = pd.read_csv(os.path.join(A.old, n + '.txt'), sep='\t', header=None, names=['s', 'e', 't'])
        a['t'] = a.t.astype(str).str.strip(); a = a[(a.e > a.s) & (a.s < len(y) / sr)].reset_index(drop=True)
        a['type'] = a.t.where(a.t.isin(['SB', 'MB', 'CRS', 'HS']), 'other')
        OLD[n] = dict(y=y, sr=sr, ann=a, dur=len(y) / sr, subj=subj(n))
    log('Oldenburg recordings', len(OLD), 'subjects', sorted({v['subj'] for v in OLD.values()}),
        'minutes %.1f' % (sum(v['dur'] for v in OLD.values()) / 60), 'events', sum(len(v['ann']) for v in OLD.values()))

OLDRS = {}
def old_bank_build(n, hop_s):
    """2 s windows over an Oldenburg recording; resample the whole recording once, then slice."""
    o = OLD[n]
    if n not in OLDRS: OLDRS[n] = {r: rs(o['y'], o['sr'], r) for r in (8000, 16000, 32000)}
    R = OLDRS[n]; nfr = int(math.ceil(o['dur'] / HOP))
    mask = frame_labels(o['ann'][['s', 'e']].values, nfr)
    starts = np.arange(0, max(o['dur'] - 2.0, 0) + 1e-9, hop_s)
    X8, W16, W32, Y = [], [], [], []
    for t0 in starts:
        y16 = fix(R[16000][int(t0 * 16000):int(t0 * 16000) + 32000], 32000); g = nf_gain(y16)
        y8 = fix(R[8000][int(t0 * 8000):int(t0 * 8000) + 16000], 16000) * g
        y32 = fix(R[32000][int(t0 * 32000):int(t0 * 32000) + 64000], 64000) * g
        X8.append(logmel8(y8)); W16.append(y16 * g); W32.append(y32)
        Y.append(fix(mask[int(round(t0 / HOP)):], NF)[:NF])
    bank = dict(X8=np.stack(X8), W16=np.stack(W16).astype(np.float32), W32=np.stack(W32).astype(np.float32),
                Y=np.stack(Y).astype(np.float32))
    return bank, starts

# ======================================================================== metrics
def to_events(b, gap=2):
    ev, on = [], None
    b = np.asarray(b)
    d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]]))
    on_i, off_i = np.where(d == 1)[0], np.where(d == -1)[0]
    merged = []
    for s, e in zip(on_i, off_i):
        if merged and s - merged[-1][1] <= gap: merged[-1][1] = e
        else: merged.append([s, e])
    return [(s * HOP, e * HOP) for s, e in merged]

def match_flags(b, gt):
    """one-to-one greedy event matching (any temporal overlap). Returns (hit flag per gt event, #pred, #pred matched)"""
    pe = to_events(b); used = set(); hit = []
    for gs, ge in gt:
        h = False
        for j, (ps, pe_) in enumerate(pe):
            if j not in used and ps < ge and pe_ > gs: used.add(j); h = True; break
        hit.append(h)
    return np.array(hit, bool), len(pe), len(used)

def clip_counts(b, gt):
    """event matching for one clip: (#pred matched, #pred, #gt matched, #gt)"""
    hit, npred, nmatch = match_flags(b, gt)
    return nmatch, npred, int(hit.sum()), len(gt)

def prf(c):
    tp_p, n_p, tp_g, n_g = c
    P = tp_p / n_p if n_p else 0.0; R = tp_g / n_g if n_g else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)

def tune_thr(prob, y):
    best = (0.5, -1)
    for t in np.linspace(0.05, 0.95, 37):
        f = f1_score(y.ravel(), (prob.ravel() > t).astype(int), zero_division=0)
        if f > best[1]: best = (t, f)
    return float(best[0])

def score_split(prob, idx, thr):
    y = Y_W[idx]; b = (prob > thr)
    C = np.array([clip_counts(b[i], EVL[j]) for i, j in enumerate(idx)]).sum(0)
    P, R, F = prf(C)
    yt, pt = y.ravel(), prob.ravel()
    return dict(frame_auroc=float(roc_auc_score(yt, pt)) if 0 < yt.mean() < 1 else np.nan,
                frame_ap=float(average_precision_score(yt, pt)),
                frame_f1=float(f1_score(yt, b.ravel(), zero_division=0)),
                event_precision=P, event_recall=R, event_f1=F, threshold=thr)

def merge_iv(iv):
    iv = sorted(iv); out = []
    for s, e in iv:
        if out and s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return np.array(out) if out else np.zeros((0, 2))

def overlaps_any(q, U):
    """q: (n,2) intervals; U: merged sorted (m,2). True if q[i] overlaps any U."""
    if len(U) == 0 or len(q) == 0: return np.zeros(len(q), bool)
    i = np.searchsorted(U[:, 0], q[:, 1], side='left') - 1     # last U starting before q end
    ok = i >= 0; i = np.clip(i, 0, len(U) - 1)
    return ok & (U[i, 1] > q[:, 0])

def one_to_one(gt, pe):
    """each annotated event needs its OWN detection (greedy, time-ordered). gt (n,2), pe (m,2) sorted, non-overlapping."""
    hit = np.zeros(len(gt), bool)
    if len(pe) == 0: return hit
    used = np.zeros(len(pe), bool); o = np.argsort(gt[:, 0])
    for i in o:
        s, e = gt[i]; j = int(np.searchsorted(pe[:, 1], s, side='right'))
        while j < len(pe) and pe[j, 0] < e:
            if not used[j]: used[j] = True; hit[i] = True; break
            j += 1
    return hit

def segment_f1(prob_b, U, dur, seg=0.1):
    """DCASE-style segment-based F1 with 100 ms segments (label-convention agnostic sanity check)."""
    ns = int(math.ceil(dur / seg)); ref = np.zeros(ns, bool); est = np.zeros(ns, bool)
    for s, e in U: ref[int(s / seg):int(math.ceil(e / seg))] = True
    f = np.where(prob_b)[0]; est[np.minimum((f * HOP / seg).astype(int), ns - 1)] = True
    tp = (ref & est).sum(); fp = (~ref & est).sum(); fn = (ref & ~est).sum()
    return float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else 0.0

def ext_metrics(prob, o, thr):
    """External event metrics. Annotation conventions differ (Oldenburg MB/CRS span many bursts), so:
    precision = detections that fall inside ANY annotated bowel-sound activity;
    recall    = annotated events that receive their OWN detection (one-to-one), so blanket detection is not rewarded."""
    ann = o['ann']; b = prob > thr
    pe = np.array(to_events(b)) if b.any() else np.zeros((0, 2))
    U = merge_iv(ann[['s', 'e']].values.tolist())
    hit_p = overlaps_any(pe, U)
    hit_g = one_to_one(ann[['s', 'e']].values, pe)
    P = hit_p.mean() if len(pe) else 0.0; R = hit_g.mean() if len(ann) else 0.0
    mask = frame_labels(U, len(prob))
    r = dict(event_precision=float(P), event_recall=float(R), event_f1=float(2 * P * R / (P + R)) if P + R else 0.0,
             segment_f1=segment_f1(b, U, o['dur']), duty_cycle=float(b.mean()),
             fa_per_min=float((~hit_p).sum() / (o['dur'] / 60)), n_pred=int(len(pe)), n_ann=int(len(ann)),
             frame_auroc=float(roc_auc_score(mask, prob)) if 0 < mask.mean() < 1 else np.nan, threshold=thr)
    for t in ['SB', 'MB', 'CRS', 'HS']:
        m = (ann.type == t).values
        r[f'recall_{t}'] = float(hit_g[m].mean()) if m.any() else np.nan
        r[f'n_{t}'] = int(m.sum())
    return r

# ======================================================================== models
import torch, torch.nn as nn, torch.nn.functional as Fnn
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
log('device', DEV, torch.cuda.get_device_name(0) if DEV == 'cuda' else '')

class CRNN(nn.Module):
    def __init__(s, nm=N_MELS, ch=64, hid=96):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                            nn.MaxPool2d((1, 2)), nn.Dropout(0.2))
        s.cnn = nn.Sequential(blk(1, ch // 2), blk(ch // 2, ch), blk(ch, ch), blk(ch, ch))
        s.rnn = nn.GRU(ch * (nm // 16), hid, batch_first=True, bidirectional=True); s.out = nn.Linear(2 * hid, 1)
        s.register_buffer('mu', torch.zeros(nm)); s.register_buffer('sd', torch.ones(nm))
    def forward(s, x):
        x = (x - s.mu) / s.sd
        if s.training:
            x = x + (torch.rand(len(x), 1, 1, device=x.device) - 0.5) * 2.0          # gain jitter (+-~4 dB)
            f0 = np.random.randint(0, N_MELS - 8); x = x.clone(); x[:, :, f0:f0 + np.random.randint(0, 8)] = 0
        h = s.cnn(x.unsqueeze(1)).permute(0, 2, 1, 3).flatten(2)
        h, _ = s.rnn(h); return s.out(h).squeeze(-1)

class ConvBlock(nn.Module):                      # identical layout to PANNs so weights load by name
    def __init__(s, i, o):
        super().__init__()
        s.conv1 = nn.Conv2d(i, o, 3, 1, 1, bias=False); s.conv2 = nn.Conv2d(o, o, 3, 1, 1, bias=False)
        s.bn1 = nn.BatchNorm2d(o); s.bn2 = nn.BatchNorm2d(o)
    def forward(s, x, pool):
        x = Fnn.relu_(s.bn1(s.conv1(x))); x = Fnn.relu_(s.bn2(s.conv2(x)))
        return Fnn.avg_pool2d(x, pool)

MEL32 = torch.tensor(librosa.filters.mel(sr=32000, n_fft=1024, n_mels=64, fmin=50, fmax=14000), dtype=torch.float32)
class CNN14F(nn.Module):
    def __init__(s, ckpt=None, hid=128):
        super().__init__()
        s.register_buffer('mel', MEL32.clone()); s.register_buffer('win', torch.hann_window(1024))
        s.bn0 = nn.BatchNorm2d(64)
        s.conv_block1 = ConvBlock(1, 64); s.conv_block2 = ConvBlock(64, 128)
        s.conv_block3 = ConvBlock(128, 256); s.conv_block4 = ConvBlock(256, 512)
        s.rnn = nn.GRU(512 * 4, hid, batch_first=True, bidirectional=True); s.out = nn.Linear(2 * hid, 1)
        s.n_loaded = 0
        if ckpt is not None:
            sd = torch.load(ckpt, map_location='cpu'); sd = sd.get('model', sd)
            own = s.state_dict(); ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
            s.load_state_dict(ok, strict=False); s.n_loaded = len(ok)
    def backbone_params(s): return [p for n, p in s.named_parameters() if n.startswith(('bn0', 'conv_block'))]
    def forward(s, w):
        with torch.autocast(device_type='cuda', enabled=False):
            S = torch.stft(w.float(), 1024, 320, 1024, s.win, center=True, return_complex=True).abs() ** 2
            x = 10 * torch.log10(torch.clamp(s.mel @ S, min=1e-10))               # (B,64,T)
        x = x[:, :, :NF].transpose(1, 2).unsqueeze(1)                             # (B,1,T,64)
        if s.training:
            f0 = np.random.randint(0, 56); x = x.clone(); x[..., f0:f0 + np.random.randint(0, 8)] = -100.0
        x = s.bn0(x.transpose(1, 3)).transpose(1, 3)
        for i, b in enumerate([s.conv_block1, s.conv_block2, s.conv_block3, s.conv_block4]):
            x = b(x, (1, 2)); x = Fnn.dropout(x, 0.2, s.training)                # frequency-only pooling
        h = x.permute(0, 2, 1, 3).flatten(2)                                     # (B,T,512*4)
        h, _ = s.rnn(h); return s.out(h).squeeze(-1)

class HubertDet(nn.Module):
    def __init__(s, name):
        super().__init__()
        from transformers import HubertModel
        s.h = HubertModel.from_pretrained(name, cache_dir=A.cache)
        try: s.h.feature_extractor._freeze_parameters()
        except Exception:
            for p_ in s.h.feature_extractor.parameters(): p_.requires_grad = False
        d = s.h.config.hidden_size
        s.out = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
    def backbone_params(s): return [p for p in s.h.parameters() if p.requires_grad]
    def forward(s, w):
        w = (w - w.mean(1, keepdim=True)) / (w.std(1, keepdim=True) + 1e-6)
        hs = s.h(w).last_hidden_state                                            # (B,~99,768)
        lg = s.out(hs).squeeze(-1)
        return Fnn.interpolate(lg.unsqueeze(1), size=NF, mode='linear', align_corners=False).squeeze(1)

INPUT = dict(crnn='X8', cnn14='W32', hubert='W16')
CNN14_CKPT = None
def get_cnn14_ckpt():
    global CNN14_CKPT
    if CNN14_CKPT: return CNN14_CKPT
    fp = os.path.join(A.cache, 'Cnn14.pth')
    if not os.path.exists(fp) or os.path.getsize(fp) < 1e8:
        subprocess.run(['curl', '-sL', '--retry', '3', '-o', fp, A.cnn14_url])
    CNN14_CKPT = fp; return fp

def build(kind):
    if kind == 'crnn': return CRNN()
    if kind == 'cnn14':
        m = CNN14F(get_cnn14_ckpt()); log('  cnn14 pretrained tensors loaded:', m.n_loaded); return m
    if kind == 'hubert': return HubertDet(A.hubert_name)

def predict(net, bank, idx, kind, bs=64):
    net.eval(); out = []
    key = INPUT[kind]
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=DEV == 'cuda'):
        for i in range(0, len(idx), bs):
            x = torch.from_numpy(bank[key][idx[i:i + bs]]).to(DEV)
            out.append(torch.sigmoid(net(x).float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, NF), np.float32)

def fit_nn(kind, bank, tr, va, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    net = build(kind).to(DEV)
    if kind == 'crnn':
        X = bank['X8'][tr]; net.mu.copy_(torch.tensor(X.mean((0, 1)))); net.sd.copy_(torch.tensor(X.std((0, 1)) + 1e-5))
        groups = [{'params': net.parameters(), 'lr': 2e-3}]
    else:
        bb = set(id(q) for q in net.backbone_params())
        head = [q for q in net.parameters() if id(q) not in bb and q.requires_grad]
        groups = [{'params': list(net.backbone_params()), 'lr': 1e-4 if kind == 'cnn14' else 3e-5},
                  {'params': head, 'lr': 1e-3}]
    E = EP[kind]; bs = 16 if kind == 'hubert' else 32
    opt = torch.optim.AdamW(groups, weight_decay=1e-3)
    steps = E * math.ceil(len(tr) / bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, [g['lr'] for g in groups], total_steps=steps + 1, pct_start=min(0.3, max(0.1, 3.0 / steps)))
    Ytr = bank['Y'][tr]; pw = torch.tensor(min(10.0, (1 - Ytr.mean()) / max(Ytr.mean(), 1e-3)), device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    scaler = torch.amp.GradScaler('cuda', enabled=DEV == 'cuda')
    key = INPUT[kind]; best, state = -1, None
    for ep in range(E):
        net.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            x = torch.from_numpy(bank[key][b]).to(DEV); y = torch.from_numpy(bank['Y'][b]).to(DEV)
            if kind != 'crnn': x = x * torch.from_numpy(10 ** (np.random.uniform(-0.3, 0.3, (len(b), 1)) )).float().to(DEV)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=DEV == 'cuda'):
                loss = lossf(net(x).float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 5)
            scaler.step(opt); scaler.update(); sch.step()
        pv = predict(net, bank, va, kind)
        ap = average_precision_score(bank['Y'][va].ravel(), pv.ravel()) if bank['Y'][va].sum() else 0
        if ap > best: best, state = ap, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state); net.eval()
    return net

# ---- LightGBM (phase 1)
def ctx_features(x):
    d = np.diff(x, axis=1, prepend=x[:, :1]); k = 5
    pad = np.pad(x, ((0, 0), (k, k), (0, 0)), mode='edge')
    win = np.stack([pad[:, i:i + NF] for i in range(2 * k + 1)], 0)
    feats = [x, d, win.mean(0), win.std(0), x - win.mean(0), x.mean(2, keepdims=True)]
    return np.concatenate(feats, 2).reshape(-1, x.shape[2] * 5 + 1)
class LGBMWrap:
    def __init__(s, m): s.m = m
def fit_lgbm(bank, tr, va, seed):
    import lightgbm as lgb
    pos = bank['Y'][tr].mean()
    m = lgb.LGBMClassifier(n_estimators=50 if A.quick else 600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.5, scale_pos_weight=(1 - pos) / max(pos, 1e-3),
                           random_state=seed, verbose=-1, n_jobs=os.cpu_count())
    m.fit(ctx_features(bank['X8'][tr]), bank['Y'][tr].ravel()); return LGBMWrap(m)
def predict_any(model, kind, bank, idx):
    if len(idx) == 0: return np.zeros((0, NF), np.float32)
    if kind == 'lgbm': return model.m.predict_proba(ctx_features(bank['X8'][idx]))[:, 1].reshape(len(idx), NF).astype(np.float32)
    return predict(model, bank, idx, kind)
def fit_any(kind, bank, tr, va, seed):
    return fit_lgbm(bank, tr, va, seed) if kind == 'lgbm' else fit_nn(kind, bank, tr, va, seed)

# ======================================================================== external inference
OLDB = {}
def old_bank(n):
    if n not in OLDB: OLDB[n] = old_bank_build(n, 1.0)
    return OLDB[n]
def stitch(n, prob, starts):
    o = OLD[n]; nfr = int(math.ceil(o['dur'] / HOP)); acc = np.zeros(nfr); cnt = np.zeros(nfr)
    for p_, s in zip(prob, starts):
        f0 = int(round(s / HOP)); f1 = min(nfr, f0 + NF)
        acc[f0:f1] += p_[:f1 - f0]; cnt[f0:f1] += 1
    return (acc / np.maximum(cnt, 1)).astype(np.float32)

# ======================================================================== main experiment
ROWS, EXT_ROWS, LOSO_ROWS = [], [], []
def save_tables():
    pd.DataFrame(ROWS).to_csv(os.path.join(A.out, 'internal_folds.csv'), index=False)
    pd.DataFrame(EXT_ROWS).to_csv(os.path.join(A.out, 'external.csv'), index=False)
    pd.DataFrame(LOSO_ROWS).to_csv(os.path.join(A.out, 'loso.csv'), index=False)

if not A.analyze_only:
    t = time.time()
    WB = make_bank([sf.read(os.path.join(A.data, s + '.wav'), dtype='float32')[0] for s in meta.stem],
                   [sf.info(os.path.join(A.data, s + '.wav')).samplerate for s in meta.stem],
                   [frame_labels(e) for e in EVL])
    Y_W = WB['Y']; log('Warsaw bank built %.1f min' % ((time.time() - t) / 60), 'pos rate %.3f' % Y_W.mean())
    if OLD:
        t = time.time()
        for n in OLD: old_bank(n)
        log('Oldenburg windows built %.1f min' % ((time.time() - t) / 60), {n: len(OLDB[n][1]) for n in OLD})
    idx = np.arange(len(meta)); groups = meta.rec.values

    def splits(protocol, seed):
        if protocol == 'grouped': return list(GroupKFold(A.folds if not A.quick else 3).split(idx, groups=groups))
        if protocol == 'random': return list(KFold(A.folds if not A.quick else 3, shuffle=True, random_state=seed).split(idx))
        if protocol == 'official': return [(idx[meta.split == 'train'], idx[meta.split == 'test'])]

    def inner(tr, protocol, seed):
        if protocol == 'random':
            a, b = next(ShuffleSplit(1, test_size=0.15, random_state=seed).split(tr))
        else:
            a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(tr, groups=groups[tr]))
        return tr[a], tr[b]

    for seed in SEEDS:
        for kind in MODELS:
            for protocol in (['grouped', 'random', 'official'] if seed == 0 else ['grouped']):
                tag = f'{kind}_{protocol}_s{seed}'
                if os.path.exists(os.path.join(A.out, f'oof_{tag}.npz')): log('skip existing', tag); continue
                try:
                    OOF = np.full((len(meta), NF), np.nan, np.float32); THR = np.full(len(meta), np.nan, np.float32)
                    ext_acc = {n: [] for n in OLD}; thrs = []
                    for k, (tr, te) in enumerate(splits(protocol, seed)):
                        t = time.time(); fi, vi = inner(tr, protocol, seed)
                        mdl = fit_any(kind, WB, fi, vi, seed)
                        thr = tune_thr(predict_any(mdl, kind, WB, vi), Y_W[vi]); thrs.append(thr)
                        pt = predict_any(mdl, kind, WB, te); OOF[te] = pt; THR[te] = thr
                        r = score_split(pt, te, thr)
                        r.update(model=kind, protocol=protocol, fold=k, seed=seed, n_train=len(fi), n_test=len(te),
                                 fit_minutes=round((time.time() - t) / 60, 2)); ROWS.append(r)
                        log(tag, 'fold', k, {q: round(v, 4) for q, v in r.items() if isinstance(v, float)})
                        if protocol == 'grouped':
                            for n in OLD:
                                bk, st = old_bank(n); ext_acc[n].append(stitch(n, predict_any(mdl, kind, bk, np.arange(len(st))), st))
                        del mdl
                        if DEV == 'cuda': torch.cuda.empty_cache()
                        save_tables()
                    np.savez_compressed(os.path.join(A.out, f'oof_{tag}.npz'), oof=OOF, thr=THR)
                    if protocol == 'grouped' and OLD:
                        thr_e = float(np.median(thrs)); ext_save = {}
                        for n in OLD:
                            pe = np.mean(ext_acc[n], 0); ext_save[n] = pe.astype(np.float16)
                            r = ext_metrics(pe, OLD[n], thr_e)
                            r.update(model=kind, seed=seed, recording=n, subject=OLD[n]['subj'], minutes=OLD[n]['dur'] / 60)
                            EXT_ROWS.append(r)
                        np.savez_compressed(os.path.join(A.out, f'ext_{tag}.npz'), **ext_save)
                        e = pd.DataFrame([r for r in EXT_ROWS if r['model'] == kind and r['seed'] == seed])
                        log(tag, 'EXTERNAL', 'F1 %.3f P %.3f R %.3f FA/min %.1f' % (e.event_f1.mean(), e.event_precision.mean(),
                                                                                     e.event_recall.mean(), e.fa_per_min.mean()))
                        save_tables()
                except Exception as ex:
                    import traceback; log('FAILED', tag, repr(ex)); log(traceback.format_exc()[-1500:])

    # ---------------- few-subject adaptation: leave-one-subject-out on Oldenburg
    if OLD:
        subs = sorted({v['subj'] for v in OLD.values()})
        TRB = {}                                                     # non-overlapping 2 s training windows
        for n in OLD:
            TRB[n] = old_bank_build(n, 2.0)[0]
        for kind in [m for m in A.loso_models.split(',') if m]:
            for seed in range(1 if A.quick else A.loso_seeds):
                for held in subs:
                    for variant in ['warsaw+old', 'old_only']:
                        try:
                            t = time.time()
                            trn = [n for n in OLD if OLD[n]['subj'] != held]
                            ob = {k: np.concatenate([TRB[n][k] for n in trn]) for k in WB}
                            og = np.concatenate([[f'old_{n}'] * len(TRB[n]['Y']) for n in trn])
                            if variant == 'warsaw+old':
                                bank = cat_banks(WB, np.arange(len(meta)), ob, np.arange(len(og)))
                                g = np.concatenate([groups, og])
                            else:
                                bank, g = ob, og
                            all_i = np.arange(len(g))
                            a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(all_i, groups=g))
                            mdl = fit_any(kind, bank, all_i[a], all_i[b], seed)
                            thr = tune_thr(predict_any(mdl, kind, bank, all_i[b]), bank['Y'][all_i[b]])
                            for n in [n for n in OLD if OLD[n]['subj'] == held]:
                                bk, st = old_bank(n)
                                pe = stitch(n, predict_any(mdl, kind, bk, np.arange(len(st))), st)
                                r = ext_metrics(pe, OLD[n], thr)
                                r.update(model=kind, variant=variant, seed=seed, held_subject=held, recording=n,
                                         fit_minutes=round((time.time() - t) / 60, 2)); LOSO_ROWS.append(r)
                                log('LOSO', kind, variant, 'held', held, n, 'F1 %.3f P %.3f R %.3f' % (r['event_f1'], r['event_precision'], r['event_recall']))
                            del mdl
                            if DEV == 'cuda': torch.cuda.empty_cache()
                            save_tables()
                        except Exception as ex:
                            import traceback; log('FAILED LOSO', kind, variant, held, repr(ex)); log(traceback.format_exc()[-1500:])
    save_tables()

# ======================================================================== analysis
def analyze():
    Y = np.stack([frame_labels(e) for e in EVL])
    rec_codes, rec_idx = np.unique(meta.rec.values, return_inverse=True)
    rng = np.random.default_rng(0); B = 300 if A.quick else 1000
    boots = [rng.integers(0, len(rec_codes), len(rec_codes)) for _ in range(B)]

    def per_rec_counts(oof, thr):
        c = np.zeros((len(rec_codes), 4))
        for i in range(len(meta)):
            if np.isnan(oof[i, 0]): continue
            c[rec_idx[i]] += clip_counts(oof[i] > thr[i], EVL[i])
        return c
    def f1c(c): return prf(c.sum(0))[2]
    def boot_f1(c): return np.array([f1c(c[b]) for b in boots])

    pooled, counts = [], {}
    for fp in sorted(glob.glob(os.path.join(A.out, 'oof_*.npz'))):
        tag = os.path.basename(fp)[4:-4]; kind, protocol, s = tag.rsplit('_', 2)
        z = np.load(fp); oof, thr = z['oof'], z['thr']
        m = ~np.isnan(oof[:, 0])
        c = per_rec_counts(oof, thr); counts[tag] = c; bf = boot_f1(c)
        P, R, F = prf(c.sum(0))
        yt, pt = Y[m].ravel(), oof[m].ravel()
        pooled.append(dict(model=kind, protocol=protocol, seed=int(s[1:]), n_clips=int(m.sum()),
                           event_f1=F, event_f1_lo=np.percentile(bf, 2.5), event_f1_hi=np.percentile(bf, 97.5),
                           event_precision=P, event_recall=R,
                           frame_f1=f1_score(yt, (oof[m] > thr[m][:, None]).ravel(), zero_division=0),
                           frame_auroc=roc_auc_score(yt, pt), frame_ap=average_precision_score(yt, pt)))
    PO = pd.DataFrame(pooled); PO.to_csv(os.path.join(A.out, 'internal_pooled_ci.csv'), index=False)

    # paired comparisons on grouped protocol (same seed): model vs crnn ; leakage inflation random - grouped
    comp = []
    for tag, c in counts.items():
        kind, protocol, s = tag.rsplit('_', 2)
        if protocol == 'grouped' and kind != 'crnn' and f'crnn_grouped_{s}' in counts:
            cb = counts[f'crnn_grouped_{s}']; d = np.array([f1c(c[b]) - f1c(cb[b]) for b in boots])
            comp.append(dict(comparison=f'{kind} - crnn', protocol='grouped', seed=s, delta_f1=f1c(c) - f1c(cb),
                             lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), p_le0=float((d <= 0).mean())))
        if protocol == 'random' and f'{kind}_grouped_{s}' in counts:
            cg = counts[f'{kind}_grouped_{s}']; d = np.array([f1c(c[b]) - f1c(cg[b]) for b in boots])
            comp.append(dict(comparison=f'{kind}: random - grouped (leakage inflation)', protocol='random-grouped', seed=s,
                             delta_f1=f1c(c) - f1c(cg), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                             p_le0=float((d <= 0).mean())))
    pd.DataFrame(comp).to_csv(os.path.join(A.out, 'paired_comparisons.csv'), index=False)

    # error analysis (grouped, all seeds): recall by Warsaw category and by event duration
    ea = []
    for fp in sorted(glob.glob(os.path.join(A.out, 'oof_*_grouped_*.npz'))):
        tag = os.path.basename(fp)[4:-4]; kind = tag.split('_')[0]; z = np.load(fp); oof, thr = z['oof'], z['thr']
        for i in range(len(meta)):
            if np.isnan(oof[i, 0]) or len(WEV[i]) == 0: continue
            hit = match_flags(oof[i] > thr[i], WEV[i][['start', 'end']].values)[0]
            for (_, ev), h in zip(WEV[i].iterrows(), hit):
                ea.append(dict(model=kind, tag=tag, category=str(ev.category), dur_ms=1000 * (ev.end - ev.start), hit=bool(h)))
    EA = pd.DataFrame(ea)
    if len(EA):
        EA['dur_bin'] = pd.cut(EA.dur_ms, [0, 15, 25, 40, 80, 1e9], labels=['<15', '15-25', '25-40', '40-80', '>80'])
        EA.groupby(['model', 'category']).hit.agg(['mean', 'size']).round(3).to_csv(os.path.join(A.out, 'recall_by_category.csv'))
        EA.groupby(['model', 'dur_bin']).hit.agg(['mean', 'size']).round(3).to_csv(os.path.join(A.out, 'recall_by_duration.csv'))

    # summaries
    S = []
    if len(PO):
        S.append('INTERNAL (pooled out-of-fold, recording-bootstrap 95% CI; grouped averaged over seeds)\n' +
                 PO.groupby(['protocol', 'model'])[['event_f1', 'event_f1_lo', 'event_f1_hi', 'event_precision', 'event_recall',
                                                     'frame_f1', 'frame_auroc']].mean().round(3).to_string())
    if os.path.exists(os.path.join(A.out, 'paired_comparisons.csv')) and len(comp):
        S.append('PAIRED COMPARISONS\n' + pd.DataFrame(comp).round(3).to_string(index=False))
    ex = pd.read_csv(os.path.join(A.out, 'external.csv')) if os.path.exists(os.path.join(A.out, 'external.csv')) and os.path.getsize(os.path.join(A.out, 'external.csv')) > 5 else pd.DataFrame()
    if len(ex):
        cols = ['event_f1', 'event_precision', 'event_recall', 'segment_f1', 'duty_cycle', 'fa_per_min', 'frame_auroc', 'recall_SB', 'recall_MB', 'recall_CRS', 'recall_HS']
        per_seed = ex.groupby(['model', 'seed'])[cols].mean()
        S.append('EXTERNAL (Oldenburg, trained on Warsaw only; mean over 7 recordings, then mean/SD over seeds)\n' +
                 per_seed.groupby('model').agg(['mean', 'std']).round(3).to_string())
        S.append('EXTERNAL per subject (mean over seeds)\n' + ex.groupby(['model', 'subject'])[['event_f1', 'event_recall', 'fa_per_min']].mean().round(3).to_string())
    lo = pd.read_csv(os.path.join(A.out, 'loso.csv')) if os.path.exists(os.path.join(A.out, 'loso.csv')) and os.path.getsize(os.path.join(A.out, 'loso.csv')) > 5 else pd.DataFrame()
    if len(lo):
        S.append('LOSO ADAPTATION (held-out Oldenburg subject; mean over recordings & seeds)\n' +
                 lo.groupby(['model', 'variant'])[['event_f1', 'event_precision', 'event_recall', 'segment_f1', 'fa_per_min', 'recall_SB']].agg(['mean', 'std']).round(3).to_string())
    if len(EA):
        S.append('WARSAW RECALL BY EVENT DURATION (grouped)\n' + EA.groupby(['model', 'dur_bin']).hit.mean().unstack().round(3).to_string())
    open(os.path.join(A.out, 'SUMMARY.txt'), 'w').write('\n\n'.join(S))
    log('\n' + '\n\n'.join(S))

try: analyze()
except Exception as ex:
    import traceback; log('ANALYSIS FAILED', repr(ex)); log(traceback.format_exc()[-2000:])
log('DONE')
