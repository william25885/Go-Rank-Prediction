import os, re, json, pickle, argparse, math, random, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore", category=UserWarning)

GAME_RE = re.compile(r'^Game\s+(\d+):', re.IGNORECASE)
MOVE_RE = re.compile(r'^[BW]\[[A-T][0-9]{1,2}\]$')
NUM_RE  = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?%?')

def _to_nums(line: str):
    out = []
    for tok in NUM_RE.findall(line):
        if tok.endswith('%'):
            tok = tok[:-1]
        try:
            out.append(float(tok))
        except:
            pass
    return out

MOVE_FEATURE_NAMES = (
    [f'policy_{i}' for i in range(1,10)] +
    [f'value_{i}'  for i in range(1,10)] +
    [f'rankp_{i}'  for i in range(1,10)] +
    ['strength', 'winrate', 'lead', 'uncertainty']   
)

def _base_frame_to_step_matrix(df_moves: pd.DataFrame) -> np.ndarray:
    arr_base = df_moves[[c for c in MOVE_FEATURE_NAMES]].to_numpy(dtype=np.float32, copy=False)

    def _max_ent(frame, base):
        a = frame[[f'{base}_{i}' for i in range(1,10)]].to_numpy(dtype=np.float32, copy=False)
        s = a.sum(axis=1, keepdims=True) + 1e-12
        p = a / s
        ent = -(p * np.log(p + 1e-12)).sum(axis=1, keepdims=True)
        mx  = a.max(axis=1, keepdims=True)
        return mx, ent

    pmax, pent = _max_ent(df_moves, 'policy')
    vmax, vent = _max_ent(df_moves, 'value')
    rkmax, rkent = _max_ent(df_moves, 'rankp')

    derived6 = np.concatenate([pmax, pent, vmax, vent, rkmax, rkent], axis=1)   
    stat37   = np.concatenate([arr_base, derived6], axis=1)                      
    d1 = np.vstack([np.zeros((1, stat37.shape[1]), dtype=np.float32),
                    np.diff(stat37, axis=0)]).astype(np.float32)             

    key_idx = [27, 28, 29]
    arr_key = arr_base[:, key_idx]
    d2 = np.vstack([np.zeros((2,3), dtype=np.float32),
                    np.diff(arr_key, n=2, axis=0)]).astype(np.float32)        

    color_is_black = (df_moves['color'].values == 'B').astype('float32')[:, None]
    pos_norm = (df_moves['move_idx'].values / max(1, df_moves['move_idx'].max())).astype('float32')[:, None]

    step = np.concatenate([arr_base, derived6, d1, d2, color_is_black, pos_norm], axis=1)  
    return step

def parse_train_file_to_moves(fpath: Path, d_label: int):
    with fpath.open('r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    i = 0
    out = []
    while i < len(lines):
        m = GAME_RE.match(lines[i])
        if not m:
            i += 1
            continue
        gid = int(m.group(1)); i += 1
        move_idx, rows = 0, []
        while i < len(lines) and not GAME_RE.match(lines[i]):
            mv = lines[i]
            if MOVE_RE.match(mv):
                color = mv[0]; move_idx += 1
                vecs, j = [], i + 1
                while j < len(lines) and len(vecs) < 5:
                    cand = _to_nums(lines[j])
                    if len(cand) in (1,3,9):
                        vecs.append(cand)
                    j += 1
                i = j
                nine  = [v for v in vecs if len(v)==9]
                ones  = [v for v in vecs if len(v)==1]
                three = [v for v in vecs if len(v)==3]
                if len(nine) < 3 or len(ones) < 1 or len(three) < 1:
                    continue
                policy, value, rankp = nine[0], nine[1], nine[2]
                strength = ones[0][0]
                winrate, lead, uncert = three[0]
                if color == 'W':
                    winrate = 1.0 - winrate
                    lead = -lead
                rows.append({
                    'move_idx': move_idx, 'color': color,
                    **{f'policy_{k+1}': policy[k] for k in range(9)},
                    **{f'value_{k+1}':  value[k]  for k in range(9)},
                    **{f'rankp_{k+1}':  rankp[k]  for k in range(9)},
                    'strength': strength, 'winrate': winrate, 'lead': lead, 'uncertainty': uncert,
                })
            else:
                i += 1
        if rows:
            df = pd.DataFrame(rows)
            out.append((f"{d_label}D_{gid}", d_label, df))
    return out

def load_all_train(train_dir: Path, workers: int=4):
    files = []
    for d in range(1,10):
        p = train_dir / f'log_{d}D_policy_train.txt'
        if p.exists():
            files.append((d, p))
        else:
            print(f"缺檔：{p}")
    if not files:
        raise RuntimeError("找不到任何訓練檔案")
    items = []
    max_proc = max(1, min(workers, len(files), os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=max_proc) as ex:
        futs = {ex.submit(parse_train_file_to_moves, p, d): (d,p) for (d,p) in files}
        for fut in as_completed(futs):
            d,p = futs[fut]; seqs = fut.result()
            items.extend(seqs)
            print(f"{p.name}: {len(seqs)}")
    if not items:
        raise RuntimeError("解析結果為空")
    return items 

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class SeqDS(Dataset):
    def __init__(self, items, le:LabelEncoder, max_len:int, mean=None, std=None, crop='rand',
                 time_cutout_prob=0.5, time_cutout_ratio=0.2, jitter_std=0.01, train=True):
        self.items = items
        self.le = le
        self.max_len = max_len
        self.crop = crop
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32).reshape(1,-1)
        self.std  = None if std  is None else np.asarray(std,  dtype=np.float32).reshape(1,-1)
        self.train = train
        self.time_cutout_prob = time_cutout_prob
        self.time_cutout_ratio = time_cutout_ratio
        self.jitter_std = jitter_std

    def __len__(self): return len(self.items)

    def _center_crop_pad(self, x):
        t = x.shape[0]
        if t > self.max_len:
            if self.crop == 'rand':
                s = np.random.randint(0, t - self.max_len + 1)
            elif self.crop == 'head':
                s = 0
            elif self.crop == 'tail':
                s = t - self.max_len
            else:
                s = max(0, (t - self.max_len)//2)
            x = x[s:s+self.max_len]
            t = self.max_len
        out = np.zeros((self.max_len, x.shape[1]), dtype=np.float32)
        take = min(t, self.max_len)
        out[:take] = x[:take]
        return out

    def _augment(self, x):
 
        if self.train and np.random.rand() < self.time_cutout_prob:
            T = x.shape[0]
            cut = max(1, int(T * self.time_cutout_ratio))
            s = np.random.randint(0, max(1, T - cut + 1))
            x[s:s+cut] = 0.0
      
        if self.train and self.jitter_std > 0:
            x = x + np.random.normal(0.0, self.jitter_std, size=x.shape).astype(np.float32)
        return x

    def __getitem__(self, idx):
        gid, lab, df_moves = self.items[idx]
        step = _base_frame_to_step_matrix(df_moves) 
        x = self._center_crop_pad(step).astype(np.float32)
        if self.train:
            x = self._augment(x)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-6)
        y = int(self.le.transform([lab])[0])
        m = (np.sum(x!=0, axis=1) > 0).astype(np.bool_)
        if not m.any():  
            m[0] = True
        return gid, torch.from_numpy(x), torch.tensor(y,dtype=torch.long), torch.from_numpy(m)

def compute_norm(items, feat_dim, max_len):
    s = np.zeros(feat_dim, dtype=np.float64); ss = np.zeros(feat_dim, dtype=np.float64); n = 0
    for _,_,df in items:
        step = _base_frame_to_step_matrix(df)
        x = step[:max_len]
        s += x.sum(axis=0); ss += (x*x).sum(axis=0); n += x.shape[0]
    mean = s / max(1,n); var = ss / max(1,n) - mean*mean; std = np.sqrt(np.maximum(var, 1e-8))
    return mean.astype('float32'), std.astype('float32')

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().cpu() for k,v in model.state_dict().items()}
    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v.detach().cpu(), alpha=1.0-self.decay)
    def state_dict(self): return self.shadow

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)
    def forward(self, x):
        pe = self.pe[:, :x.size(1), :].to(dtype=x.dtype)  
        return x + pe

class BiLSTMClf(nn.Module):
    def __init__(self, in_dim, hidden=224, layers=2, dropout=0.25, num_class=9):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, dropout=dropout,
                            bidirectional=True, batch_first=True)
        self.att  = nn.Linear(hidden*2, 1)
        self.drop = nn.Dropout(dropout)
        self.head_ce   = nn.Linear(hidden*2, num_class)
        self.head_ord  = nn.Linear(hidden*2, num_class-1)
    def forward(self, x, mask):
        h = torch.relu(self.proj(x))
        out,_ = self.lstm(h)
        la = self.att(out).squeeze(-1)
        neg_inf = torch.finfo(la.dtype).min  
        la = la.masked_fill(~mask, neg_inf)
        w = torch.softmax(la, dim=1)
        pooled = torch.sum(out*w.unsqueeze(-1), dim=1)
        z = self.drop(pooled)
        return {'ce_logits': self.head_ce(z), 'ord_logits': self.head_ord(z)}

class TinyTransformer(nn.Module):
    def __init__(self, in_dim, d_model=224, nhead=7, layers=3, dim_ff=640, dropout=0.15, num_class=9):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos  = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                         dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head_ce   = nn.Linear(d_model, num_class)
        self.head_ord  = nn.Linear(d_model, num_class-1)
    def forward(self, x, mask):
        h = self.pos(self.proj(x))
        enc = self.encoder(h, src_key_padding_mask=~mask)  
        m = mask.float().unsqueeze(-1)
        pooled = (enc*m).sum(1)/(m.sum(1)+1e-6)
        z = self.drop(self.norm(pooled))
        return {'ce_logits': self.head_ce(z), 'ord_logits': self.head_ord(z)}

@torch.no_grad()
def coral_logits_to_proba(logits: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(logits)
    B,K1 = s.shape; K = K1+1
    p = torch.zeros(B,K, device=logits.device, dtype=logits.dtype)
    p[:,0] = 1 - s[:,0]
    for j in range(1,K-1):
        p[:,j] = s[:,j-1] - s[:,j]
    p[:,K-1] = s[:,K-2]
    return p

def _ordinal_margin_loss(ord_logits: torch.Tensor, margin: float = 0.0):

    if ord_logits.size(1) <= 1:
        return ord_logits.new_tensor(0.0)
    diffs = ord_logits[:, :-1] - ord_logits[:, 1:] 
    pen = torch.relu(margin - diffs)
    return pen.mean()

def seq_fit_fold(model, train_items, val_items, le, max_len, mean, std, device,
                 epochs=28, lr=3e-4, weight_decay=1e-2, patience=7, batch_size=128,
                 label_smoothing=0.05, ord_margin=0.02):


    y_all = np.array([int(le.transform([lab])[0]) for (_,lab,_) in train_items], dtype=np.int64)
    cls_counts = pd.Series(y_all).value_counts().sort_index()
    inv = (1.0 / cls_counts).values
    inv = inv / inv.mean()

    ds_tr = SeqDS(train_items, le, max_len=max_len, mean=mean, std=std, crop='rand',
                  time_cutout_prob=0.6, time_cutout_ratio=0.25, jitter_std=0.012, train=True)
    ds_va = SeqDS(val_items,   le, max_len=max_len, mean=mean, std=std, crop='center',
                  time_cutout_prob=0.0, time_cutout_ratio=0.0, jitter_std=0.0,   train=False)

    y_tr = np.array([int(le.transform([lab])[0]) for (_,lab,_) in train_items], dtype=np.int64)
    w = torch.tensor([inv[c] for c in y_tr], dtype=torch.float32)
    sampler = torch.utils.data.WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)

    ld_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=0, pin_memory=True)
    ld_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    criterion_ce  = nn.CrossEntropyLoss(weight=torch.tensor(inv, dtype=torch.float32, device=device),
                                        label_smoothing=label_smoothing)
    criterion_bce = nn.BCEWithLogitsLoss()

    ema = EMA(model, decay=0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2)


    use_cuda = (device.type == 'cuda')
    use_bf16 = bool(use_cuda and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    try:
        GradScalerNew = torch.amp.GradScaler
    except AttributeError:
        GradScalerNew = torch.cuda.amp.GradScaler  
    scaler = GradScalerNew(enabled=(use_cuda and not use_bf16))

    try:
        autocast = torch.amp.autocast
    except AttributeError:
        from torch.cuda.amp import autocast as _old_autocast
        def autocast(device_type, dtype, enabled=True):

            return _old_autocast(enabled=enabled, dtype=dtype)

    best_acc, best_state, es = -1.0, None, 0
    model.to(device)

    for ep in range(1, epochs+1):
        model.train()
        for _, X, y, m in ld_tr:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
                out = model(X, m)
                loss_ce = criterion_ce(out['ce_logits'], y)
                K = out['ord_logits'].size(1) + 1
                y_bin = torch.zeros(y.size(0), K-1, device=y.device, dtype=out['ord_logits'].dtype)
                for k in range(K-1):
                    y_bin[:, k] = (y > k).float()
                loss_bce = criterion_bce(out['ord_logits'], y_bin)
                loss_ord_margin = _ordinal_margin_loss(out['ord_logits'], margin=ord_margin)
                loss = 0.28*loss_ce + 0.70*loss_bce + 0.02*loss_ord_margin

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()

            sch.step()
            ema.update(model)

        with torch.no_grad():
            shadow = ema.state_dict()
            cur = model.state_dict()
            model.load_state_dict(shadow, strict=True)
            model.eval()
            ys, ps = [], []
            for _, X, y, m in ld_va:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                m = m.to(device, non_blocking=True)
                out = model(X, m)
                p_ord = coral_logits_to_proba(out['ord_logits'])
                p_ce  = torch.softmax(out['ce_logits'], dim=1)
                p     = 0.7*p_ord + 0.3*p_ce
                ps.append(p.argmax(dim=1).cpu().numpy())
                ys.append(y.cpu().numpy())
            y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
            acc = float(accuracy_score(y_true, y_pred))
            model.load_state_dict(cur, strict=True)

        if acc > best_acc + 1e-6:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k,v in ema.state_dict().items()}
            es = 0
        else:
            es += 1
        print(f"epoch={ep:02d}  val_acc={acc:.4f}  best={best_acc:.4f}")
        if es >= patience:
            break
    return best_acc, best_state

@torch.no_grad()
def seq_predict_proba(model, state_dict, items, le, max_len, mean, std, device):
    ds = SeqDS(items, le, max_len=max_len, mean=mean, std=std, crop='center',
               time_cutout_prob=0.0, time_cutout_ratio=0.0, jitter_std=0.0, train=False)
    ld = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    allp = []
    for _, X, _, m in ld:
        X = X.to(device)
        m = m.to(device)
        out = model(X, m)
        p = 0.7*coral_logits_to_proba(out['ord_logits']) + 0.3*torch.softmax(out['ce_logits'], dim=1)
        allp.append(p.cpu().numpy())
    return np.vstack(allp)

def _entropy_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.maximum(arr, 0.0)
    s = arr.sum(axis=1, keepdims=True) + 1e-12
    p = arr / s
    return -(p * (np.log(p + 1e-12))).sum(axis=1)

def _safe_skew(a):
    a = a[np.isfinite(a)]
    if a.size < 3 or np.std(a) < 1e-12: return 0.0
    from scipy.stats import skew
    return float(skew(a))

def _add_stats(feats, prefix, s: pd.Series):
    x = pd.to_numeric(s, errors='coerce').astype('float64')
    x = x[np.isfinite(x)]
    if x.empty:
        for suf in ('mean','std','min','max','med','p25','p75','skew','dmean','dstd'):
            feats[f'{prefix}_{suf}'] = 0.0
        return
    feats[f'{prefix}_mean'] = float(x.mean());  feats[f'{prefix}_std'] = float(x.std())
    feats[f'{prefix}_min']  = float(x.min());   feats[f'{prefix}_max'] = float(x.max())
    feats[f'{prefix}_med']  = float(x.median())
    q25,q75 = np.percentile(x.values,[25,75])
    feats[f'{prefix}_p25']  = float(q25);       feats[f'{prefix}_p75'] = float(q75)
    feats[f'{prefix}_skew'] = _safe_skew(x.values)
    dx = x.diff().dropna()
    feats[f'{prefix}_dmean']= float(dx.mean()) if not dx.empty else 0.0
    feats[f'{prefix}_dstd'] = float(dx.std())  if not dx.empty else 0.0

def _add_pref_features(feats, prefix, df, base):
    cols = [f'{base}_{i}' for i in range(1,10)]
    if not set(cols).issubset(df.columns) or df.empty:
        feats[f'{prefix}_{base}_entropy_mean']=0.0
        feats[f'{prefix}_{base}_argmax_mode']=-1
        feats[f'{prefix}_{base}_max_mean']=0.0
        return
    arr = df[cols].to_numpy(dtype=np.float64, copy=False)
    ent = _entropy_rows(arr)
    feats[f'{prefix}_{base}_entropy_mean'] = float(ent.mean())
    am = np.argmax(arr, axis=1)
    feats[f'{prefix}_{base}_argmax_mode']  = int(np.bincount(am).argmax()) if am.size else -1
    feats[f'{prefix}_{base}_max_mean']     = float(np.max(arr, axis=1).mean())

def _split_open_mid_end(df: pd.DataFrame):
    gmax = int(df.shape[0])
    thirds = max(1, gmax//3)
    open_part = df.iloc[:thirds]
    mid_part  = df.iloc[thirds: 2*thirds]
    end_part  = df.iloc[2*thirds:]
    return open_part, mid_part, end_part

def _aggregate_one_game(df_game: pd.DataFrame) -> pd.Series:
    feats={}
    base_cols = [f'policy_{i}' for i in range(1,10)] + \
                [f'value_{i}'  for i in range(1,10)] + \
                [f'rankp_{i}'  for i in range(1,10)] + \
                ['strength','winrate','lead','uncertainty']
    for c in base_cols:
        if c in df_game.columns: _add_stats(feats, f'all_{c}', df_game[c])
    _add_pref_features(feats, 'all', df_game, 'policy')
    _add_pref_features(feats, 'all', df_game, 'value')
    _add_pref_features(feats, 'all', df_game, 'rankp')
    feats['all_n_moves']=int(df_game.shape[0])


    op, md, ed = _split_open_mid_end(df_game)
    for tag, sub in [('open',op), ('mid',md), ('end',ed)]:
        _add_pref_features(feats, tag, sub, 'policy')
        _add_pref_features(feats, tag, sub, 'value')
        _add_pref_features(feats, tag, sub, 'rankp')
        _add_stats(feats, f'{tag}_lead', sub['lead'] if 'lead' in sub.columns else pd.Series(dtype='float32'))
        _add_stats(feats, f'{tag}_uncert', sub['uncertainty'] if 'uncertainty' in sub.columns else pd.Series(dtype='float32'))
    return pd.Series(feats, dtype='float32')

def build_tabular_matrix(items):
    recs = []
    for gid, lab, df in items:
        s = _aggregate_one_game(df)
        s['gid']=gid; s['label']=lab
        recs.append(s)
    X = pd.DataFrame(recs).fillna(0.0)
    y = X.pop('label').astype(int).values
    _ = X.pop('gid')
    vt = VarianceThreshold(threshold=1e-12)
    Xv = vt.fit_transform(X.astype('float32'))
    cols = X.columns[vt.get_support(indices=True)]
    X = pd.DataFrame(Xv, columns=cols)
    return X, y, list(cols)

def fit_tabular_cv(X, y, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    try:
        from catboost import CatBoostClassifier, Pool
        print("[TAB] Use CatBoost")
        oof = np.zeros((len(y), len(np.unique(y))), dtype=np.float32)
        accs=[]
        for f,(tr,va) in enumerate(skf.split(X,y),1):
            clf = CatBoostClassifier(
                depth=8, learning_rate=0.05, l2_leaf_reg=6.0, iterations=1800,
                loss_function='MultiClass', eval_metric='Accuracy',
                random_seed=100+f, verbose=False, task_type='CPU'
            )
            clf.fit(Pool(X.iloc[tr], y[tr]), eval_set=Pool(X.iloc[va], y[va]), verbose=False)
            p = clf.predict_proba(X.iloc[va]); oof[va]=p
            acc = accuracy_score(y[va], np.argmax(p,axis=1)); accs.append(acc)
            print(f"[TAB][CV] fold{f} acc={acc:.4f}")
        mean_acc = float(np.mean(accs))
        model = clf.fit(X,y)
        return model, 'catboost', oof, mean_acc
    except Exception as e:
        print(f"[TAB] CatBoost not available -> HGBT. reason: {e}")
        from sklearn.ensemble import HistGradientBoostingClassifier
        oof = np.zeros((len(y), len(np.unique(y))), dtype=np.float32)
        accs=[]
        for f,(tr,va) in enumerate(skf.split(X,y),1):
            clf = HistGradientBoostingClassifier(max_depth=None, learning_rate=0.05,
                                                l2_regularization=1e-3, max_leaf_nodes=255,
                                                random_state=100+f)
            clf.fit(X.iloc[tr], y[tr])
            p = clf.predict_proba(X.iloc[va]); oof[va]=p
            acc = accuracy_score(y[va], np.argmax(p,axis=1)); accs.append(acc)
            print(f"[TAB][CV] fold{f} acc={acc:.4f}")
        mean_acc = float(np.mean(accs))
        model = clf.fit(X,y)
        return model, 'sk_hgbt', oof, mean_acc

def _entropy_rows_np(arr: np.ndarray) -> np.ndarray:
    arr = np.maximum(arr, 0.0)
    s = arr.sum(axis=1, keepdims=True) + 1e-12
    p = arr / s
    return -(p * (np.log(p + 1e-12))).sum(axis=1)

def compute_meta_side_features(df: pd.DataFrame) -> np.ndarray:
    n = df.shape[0]
    def _ent(base, sub=None):
        tgt = df if sub is None else sub
        cols = [f'{base}_{i}' for i in range(1,10)]
        if not set(cols).issubset(tgt.columns) or tgt.shape[0] == 0: return 0.0
        arr = tgt[cols].to_numpy(dtype=np.float32, copy=False)
        return float(_entropy_rows_np(arr).mean())
    p_ent = _ent('policy'); v_ent = _ent('value'); r_ent = _ent('rankp')
    log_moves = float(np.log1p(n))
    win_std = float(pd.to_numeric(df['winrate'], errors='coerce').astype('float64').std()) if n>0 else 0.0
    lead_abs_mean = float(np.abs(pd.to_numeric(df['lead'], errors='coerce').astype('float64')).mean()) if n>0 else 0.0
    uncert_mean = float(pd.to_numeric(df['uncertainty'], errors='coerce').astype('float64').mean()) if n>0 else 0.0
    op, md, ed = _split_open_mid_end(df)
    open_ent = _ent('policy', op)
    mid_ent  = _ent('policy', md)
    end_ent  = _ent('policy', ed)
    return np.array([p_ent, v_ent, r_ent, log_moves, win_std, lead_abs_mean, uncert_mean,
                     open_ent, mid_ent, end_ent], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_dir', type=str, default='train_set')
    ap.add_argument('--out_dir',   type=str, default='.')
    ap.add_argument('--workers',   type=int, default=max(1, (os.cpu_count() or 1)//2))
    ap.add_argument('--seq_len',   type=int, default=120)
    ap.add_argument('--epochs',    type=int, default=30)
    ap.add_argument('--batch_size',type=int, default=128)
    ap.add_argument('--lr',        type=float, default=3e-4)
    ap.add_argument('--gpu',       action='store_true')
    ap.add_argument('--seed',      type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu')

    print("解析訓練資料")
    items = load_all_train(Path(args.train_dir), workers=args.workers)
    labels = np.array([lab for (_,lab,_) in items], dtype=np.int64)
    le = LabelEncoder(); y_enc = le.fit_transform(labels)
    num_class = len(le.classes_); assert num_class>=2
    feat_dim = 79
    mean, std = compute_norm(items, feat_dim, args.seq_len)

    print("建立 5-fold")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    idx = np.arange(len(items))

    print("OOF：Transformer / BiLSTM")
    oof_tf = np.zeros((len(items), num_class), dtype=np.float32)
    oof_bl = np.zeros((len(items), num_class), dtype=np.float32)
    cv_tf, cv_bl = [], []
    folds = list(skf.split(idx, y_enc))
    for f,(tr,va) in enumerate(folds,1):
        tr_items = [items[i] for i in tr]; va_items = [items[i] for i in va]

        tf = TinyTransformer(in_dim=feat_dim, d_model=224, nhead=7, layers=3, dim_ff=640,
                             dropout=0.15, num_class=num_class).to(device)
        acc_tf, state_tf = seq_fit_fold(tf, tr_items, va_items, le, args.seq_len, mean, std, device,
                                        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                                        patience=7, label_smoothing=0.05, ord_margin=0.02)
        cv_tf.append(acc_tf)
        p_tf = seq_predict_proba(tf, state_tf, va_items, le, args.seq_len, mean, std, device)
        oof_tf[va] = p_tf

        bl = BiLSTMClf(in_dim=feat_dim, hidden=224, layers=2, dropout=0.25, num_class=num_class).to(device)
        acc_bl, state_bl = seq_fit_fold(bl, tr_items, va_items, le, args.seq_len, mean, std, device,
                                        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                                        patience=7, label_smoothing=0.05, ord_margin=0.02)
        cv_bl.append(acc_bl)
        p_bl = seq_predict_proba(bl, state_bl, va_items, le, args.seq_len, mean, std, device)
        oof_bl[va] = p_bl

        print(f"[SEQ][CV] fold{f}  tf={acc_tf:.4f}  bl={acc_bl:.4f}")

    print(f"[SEQ] TF mean={np.mean(cv_tf):.4f}  BL mean={np.mean(cv_bl):.4f}")

    print("Tabular OOF")
    X_tab, y_tab, cols_tab = build_tabular_matrix(items)
    tab_model, tab_type, oof_tab, tab_mean = fit_tabular_cv(X_tab, y_tab, n_splits=5, random_state=args.seed)

    print("準備特徵")
    side_feats = np.vstack([compute_meta_side_features(df) for _,_,df in items])  # [N,10]
    meta_X = np.concatenate([oof_tf, oof_bl, oof_tab, side_feats], axis=1)        # [N, K*3+10]
    meta_scaler = StandardScaler().fit(meta_X)
    meta_Xs = meta_scaler.transform(meta_X)
    meta = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=2.0, max_iter=400, n_jobs=None)
    meta.fit(meta_Xs, y_enc)
    y_pred = meta.predict(meta_Xs)
    stack_oof_acc = accuracy_score(y_enc, y_pred)
    print(f"[META] OOF stacking acc={stack_oof_acc:.4f}")

    print("全資料重訓")
    def train_full(seed):
        set_seed(seed)
        tf_full = TinyTransformer(in_dim=feat_dim, d_model=224, nhead=7, layers=3, dim_ff=640,
                                  dropout=0.15, num_class=num_class).to(device)
        _, tf_state = seq_fit_fold(tf_full, items, items, le, args.seq_len, mean, std, device,
                                   epochs=max(8, args.epochs//2), lr=args.lr, batch_size=args.batch_size,
                                   patience=4, label_smoothing=0.03, ord_margin=0.02)
        bl_full = BiLSTMClf(in_dim=feat_dim, hidden=224, layers=2, dropout=0.25, num_class=num_class).to(device)
        _, bl_state = seq_fit_fold(bl_full, items, items, le, args.seq_len, mean, std, device,
                                   epochs=max(8, args.epochs//2), lr=args.lr, batch_size=args.batch_size,
                                   patience=4, label_smoothing=0.03, ord_margin=0.02)
        return tf_state, bl_state
    tf_s1, bl_s1 = train_full(args.seed)
    tf_s2, bl_s2 = train_full(args.seed + 777)

    print("存檔")
    payload = {
        'label_encoder': le,
        'num_class': num_class,
        'norm': {'mean': mean, 'std': std},
        'seq_hparams': {'feat_dim':79, 'seq_len': args.seq_len,
                        'tf':{'hidden':224,'heads':7,'layers':3,'ff':640,'dropout':0.15},
                        'bl':{'hidden':224,'layers':2,'dropout':0.25}},
        'seq_states': {
            'tf': [tf_s1, tf_s2],
            'bl': [bl_s1, bl_s2]
        },
        'tabular': {
            'model_type': tab_type,
            'model': tab_model,
            'cols': cols_tab
        },
        'meta': {
            'scaler': meta_scaler,
            'clf': meta,
            'side_keys': ['policy_ent_mean','value_ent_mean','rankp_ent_mean','log1p_n_moves',
                          'winrate_std','lead_abs_mean','uncert_mean','open_ent','mid_ent','end_ent']
        },
        'stack_info': {
            'oof_acc_seq_tf': float(np.mean(cv_tf)),
            'oof_acc_seq_bl': float(np.mean(cv_bl)),
            'oof_acc_tab': float(tab_mean),
            'oof_acc_meta': float(stack_oof_acc)
        },
        'version': 5
    }
    outp = out_dir / 'model_stackx.pkl'
    with open(outp, 'wb') as f: pickle.dump(payload, f)
    (out_dir / 'train_summary.json').write_text(
        json.dumps(payload['stack_info'], indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f"[OK] Saved -> {outp}")
    print("[DONE]")

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--train_dir', 'train_set',
            '--out_dir',   '.',
            '--gpu',
            '--seq_len',   '120',
            '--epochs',    '30',
            '--batch_size','128',
            '--lr',        '0.0003',
            '--seed',      '42'
        ])
    main()
