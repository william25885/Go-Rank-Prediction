import re, pickle, argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn

GAME_RE = re.compile(r'^Game\s+(\d+):', re.IGNORECASE)
MOVE_RE = re.compile(r'^[BW]\[[A-T][0-9]{1,2}\]$')
NUM_RE  = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?%?')

def _to_nums(line: str):
    out = []
    for tok in NUM_RE.findall(line):
        if tok.endswith('%'): tok = tok[:-1]
        try: out.append(float(tok))
        except: pass
    return out

def parse_file_to_games(fpath: Path):
    with fpath.open('r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    i=0; games=[]
    while i < len(lines):
        m = GAME_RE.match(lines[i])
        if not m: i+=1; continue
        gid = int(m.group(1)); i += 1
        move_idx, rows = 0, []
        while i < len(lines) and not GAME_RE.match(lines[i]):
            mv = lines[i]
            if MOVE_RE.match(mv):
                color = mv[0]; move_idx += 1
                vecs, j = [], i + 1
                while j < len(lines) and len(vecs) < 5:
                    cand = _to_nums(lines[j])
                    if len(cand) in (1,3,9): vecs.append(cand)
                    j += 1
                i = j
                nine  = [v for v in vecs if len(v)==9]
                ones  = [v for v in vecs if len(v)==1]
                three = [v for v in vecs if len(v)==3]
                if len(nine) < 3 or len(ones) < 1 or len(three) < 1: continue
                policy, value, rankp = nine[0], nine[1], nine[2]
                strength = ones[0][0]; winrate, lead, uncert = three[0]
                if color == 'W': winrate = 1.0 - winrate; lead = -lead
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
            games.append(pd.DataFrame(rows))
    return games

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
        ent = -(p * (np.log(p + 1e-12))).sum(axis=1, keepdims=True)
        mx  = a.max(axis=1, keepdims=True)
        return mx, ent
    pmax,pent = _max_ent(df_moves,'policy')
    vmax,vent = _max_ent(df_moves,'value')
    rkmax,rkent = _max_ent(df_moves,'rankp')
    derived6 = np.concatenate([pmax,pent,vmax,vent,rkmax,rkent], axis=1)
    stat37 = np.concatenate([arr_base, derived6], axis=1)
    d1 = np.vstack([np.zeros((1, stat37.shape[1]), dtype=np.float32),
                    np.diff(stat37, axis=0)]).astype(np.float32)
    key_idx = [27,28,29]
    arr_key = arr_base[:, key_idx]
    d2 = np.vstack([np.zeros((2,3),dtype=np.float32),
                    np.diff(arr_key, n=2, axis=0)]).astype(np.float32)
    color_is_black = (df_moves['color'].values == 'B').astype('float32')[:, None]
    pos_norm = (df_moves['move_idx'].values / max(1, df_moves['move_idx'].max())).astype('float32')[:, None]
    step = np.concatenate([arr_base, derived6, d1, d2, color_is_black, pos_norm], axis=1)
    return step

def center_crop_pad(x: np.ndarray, max_len: int) -> np.ndarray:
    t = x.shape[0]
    if t > max_len:
        s = max(0, (t - max_len)//2)
        x = x[s:s+max_len]; t = max_len
    out = np.zeros((max_len, x.shape[1]), dtype=np.float32)
    take = min(t, max_len); out[:take] = x[:take]
    return out

def _entropy_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.maximum(arr, 0.0)
    s = arr.sum(axis=1, keepdims=True) + 1e-12
    p = arr / s
    return -(p * (np.log(p + 1e-12))).sum(axis=1)

def _split_open_mid_end(df: pd.DataFrame):
    gmax = int(df.shape[0])
    thirds = max(1, gmax//3)
    return df.iloc[:thirds], df.iloc[thirds: 2*thirds], df.iloc[2*thirds:]

def compute_meta_side_features(df: pd.DataFrame) -> np.ndarray:
    n = df.shape[0]
    def _ent(base, sub=None):
        tgt = df if sub is None else sub
        cols = [f'{base}_{i}' for i in range(1,10)]
        if not set(cols).issubset(tgt.columns) or tgt.shape[0] == 0: return 0.0
        arr = tgt[cols].to_numpy(dtype=np.float32, copy=False)
        return float(_entropy_rows(arr).mean())
    p_ent = _ent('policy'); v_ent = _ent('value'); r_ent = _ent('rankp')
    log_moves = float(np.log1p(n))
    win_std = float(pd.to_numeric(df['winrate'], errors='coerce').astype('float64').std()) if n>0 else 0.0
    lead_abs_mean = float(np.abs(pd.to_numeric(df['lead'], errors='coerce').astype('float64')).mean()) if n>0 else 0.0
    uncert_mean = float(pd.to_numeric(df['uncertainty'], errors='coerce').astype('float64').mean()) if n>0 else 0.0
    op, md, ed = _split_open_mid_end(df)
    open_ent = _ent('policy', op); mid_ent = _ent('policy', md); end_ent = _ent('policy', ed)
    return np.array([p_ent, v_ent, r_ent, log_moves, win_std, lead_abs_mean, uncert_mean,
                     open_ent, mid_ent, end_ent], dtype=np.float32)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

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
        h = torch.relu(self.proj(x)); out,_ = self.lstm(h)
        la = self.att(out).squeeze(-1).masked_fill(~mask, -1e9); w = torch.softmax(la, dim=1)
        pooled = torch.sum(out*w.unsqueeze(-1), dim=1); z=self.drop(pooled)
        return {'ce_logits': self.head_ce(z), 'ord_logits': self.head_ord(z)}

class TinyTransformer(nn.Module):
    def __init__(self, in_dim, d_model=224, nhead=7, layers=3, dim_ff=640, dropout=0.15, num_class=9):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model); self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                         dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(d_model); self.drop = nn.Dropout(dropout)
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
    s = torch.sigmoid(logits); B,K1 = s.shape; K = K1+1
    p = torch.zeros(B,K, device=logits.device, dtype=logits.dtype)
    p[:,0] = 1 - s[:,0]
    for j in range(1,K-1): p[:,j] = s[:,j-1] - s[:,j]
    p[:,K-1] = s[:,K-2]
    return p

@torch.no_grad()
def seq_game_proba(model, state, df, mean, std, seq_len, device):
    step = _base_frame_to_step_matrix(df).astype(np.float32)
    T = step.shape[0]
    views=[]
    if T <= seq_len:
        views.append(center_crop_pad(step, seq_len))
    else:
        
        center_s = max(0,(T-seq_len)//2)
        q1_s     = max(0,(T-seq_len)//4)
        q5_s     = max(0,(T-seq_len)*3//4)
        e1_s     = max(0,(T-seq_len)//8)
        e5_s     = max(0,(T-seq_len)*5//8)
        views.extend([
            step[:seq_len],
            step[-seq_len:],
            step[center_s:center_s+seq_len],
            step[q1_s:q1_s+seq_len],
            step[e1_s:e1_s+seq_len],
            step[q5_s:q5_s+seq_len],
        ])
    X = torch.from_numpy(np.stack(views,0)).to(device)               
    M = torch.ones(X.size(0), X.size(1), dtype=torch.bool, device=device)
    mean_t = torch.from_numpy(mean).view(1,1,-1).to(device)
    std_t  = torch.from_numpy(std).view(1,1,-1).to(device)
    X = (X - mean_t) / (std_t + 1e-6)

    model.load_state_dict(state, strict=True); model.to(device).eval()
    out = model(X, M)
    p = 0.7*coral_logits_to_proba(out['ord_logits']) + 0.3*torch.softmax(out['ce_logits'], dim=1)
    return p.mean(0).detach().cpu().numpy()  # (K,)

def _add_stats(feats, prefix, s: pd.Series):
    x = pd.to_numeric(s, errors='coerce').astype('float64'); x=x[np.isfinite(x)]
    if x.empty:
        for suf in ('mean','std','min','max','med','p25','p75','skew','dmean','dstd'):
            feats[f'{prefix}_{suf}']=0.0
        return
    feats[f'{prefix}_mean']=float(x.mean()); feats[f'{prefix}_std']=float(x.std())
    feats[f'{prefix}_min']=float(x.min());  feats[f'{prefix}_max']=float(x.max())
    feats[f'{prefix}_med']=float(x.median())
    q25,q75 = np.percentile(x.values,[25,75])
    feats[f'{prefix}_p25']=float(q25); feats[f'{prefix}_p75']=float(q75)
    dx = x.diff().dropna()
    feats[f'{prefix}_skew']=0.0
    feats[f'{prefix}_dmean']=float(dx.mean()) if not dx.empty else 0.0
    feats[f'{prefix}_dstd']=float(dx.std())  if not dx.empty else 0.0

def _add_pref_features(feats, prefix, df, base):
    cols = [f'{base}_{i}' for i in range(1,10)]
    if not set(cols).issubset(df.columns) or df.empty:
        feats[f'{prefix}_{base}_entropy_mean']=0.0
        feats[f'{prefix}_{base}_argmax_mode']=-1
        feats[f'{prefix}_{base}_max_mean']=0.0
        return
    arr = df[cols].to_numpy(dtype=np.float64, copy=False)
    ent = _entropy_rows(arr)
    feats[f'{prefix}_{base}_entropy_mean']=float(ent.mean())
    am = np.argmax(arr, axis=1)
    feats[f'{prefix}_{base}_argmax_mode']=int(np.bincount(am).argmax()) if am.size else -1
    feats[f'{prefix}_{base}_max_mean']=float(np.max(arr, axis=1).mean())

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
    gmax = int(df_game.shape[0]); thirds = max(1, gmax//3)
    op = df_game.iloc[:thirds]; md = df_game.iloc[thirds:2*thirds]; ed = df_game.iloc[2*thirds:]
    for tag, sub in [('open',op), ('mid',md), ('end',ed)]:
        _add_pref_features(feats, tag, sub, 'policy')
        _add_pref_features(feats, tag, sub, 'value')
        _add_pref_features(feats, tag, sub, 'rankp')
        _add_stats(feats, f'{tag}_lead', sub['lead'] if 'lead' in sub.columns else pd.Series(dtype='float32'))
        _add_stats(feats, f'{tag}_uncert', sub['uncertainty'] if 'uncertainty' in sub.columns else pd.Series(dtype='float32'))
    return pd.Series(feats, dtype='float32')


def main():
    import os
    # 獲取當前腳本所在目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_dir',   type=str, default=os.path.join(script_dir, "test_set"))
    ap.add_argument('--model_path', type=str, default=os.path.join(script_dir, "model_stackx.pkl"))
    ap.add_argument('--out_csv',    type=str, default=os.path.join(script_dir, "submission.csv"))
    args = ap.parse_args()

    payload = pickle.load(open(args.model_path, 'rb'))
    le = payload['label_encoder']; num_class = payload['num_class']
    mean = payload['norm']['mean']; std = payload['norm']['std']
    hp = payload['seq_hparams']
    tf_states = payload['seq_states']['tf']   
    bl_states = payload['seq_states']['bl']   
    tab_type  = payload['tabular']['model_type']
    tab_model = payload['tabular']['model']
    cols_tab  = payload['tabular']['cols']
    meta_scaler = payload['meta']['scaler']
    meta_clf    = payload['meta']['clf']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tf = TinyTransformer(in_dim=hp['feat_dim'], d_model=hp['tf']['hidden'], nhead=hp['tf']['heads'],
                         layers=hp['tf']['layers'], dim_ff=hp['tf']['ff'],
                         dropout=hp['tf']['dropout'], num_class=num_class)
    bl = BiLSTMClf(in_dim=hp['feat_dim'], hidden=hp['bl']['hidden'], layers=hp['bl']['layers'],
                   dropout=hp['bl']['dropout'], num_class=num_class)

    test_dir = Path(args.test_dir)
    files = sorted(test_dir.glob('*.txt'))
    if not files: raise FileNotFoundError(f"沒有 .txt：{test_dir}")

    ids, ranks = [], []
    for p in files:
        games = parse_file_to_games(p)
        if not games:
            ids.append(p.stem); ranks.append(5); continue  

        meta_ps = []
        for df in games:
            ps_tf = [seq_game_proba(tf, st, df, mean, std, hp['seq_len'], device) for st in tf_states]
            p_tf = np.mean(np.vstack(ps_tf), axis=0)
            ps_bl = [seq_game_proba(bl, st, df, mean, std, hp['seq_len'], device) for st in bl_states]
            p_bl = np.mean(np.vstack(ps_bl), axis=0)

            # Tabular
            def game_tab_proba(df_game):
                s = _aggregate_one_game(df_game)
                X = pd.DataFrame([s]).fillna(0.0)
                for c in cols_tab:
                    if c not in X.columns: X[c] = 0.0
                X = X[cols_tab]
                if hasattr(tab_model, "predict_proba"):
                    return np.array(tab_model.predict_proba(X), dtype=np.float32)[0]
                else:
                    logits = tab_model.decision_function(X)
                    e = np.exp(logits - logits.max(axis=1, keepdims=True))
                    P = e / e.sum(axis=1, keepdims=True)
                    return P[0].astype(np.float32)
            p_tb = game_tab_proba(df)

            side = compute_meta_side_features(df)
            meta_x = np.concatenate([p_tf, p_bl, p_tb, side], axis=0).reshape(1, -1)
            meta_xs = meta_scaler.transform(meta_x)
            mp = meta_clf.predict_proba(meta_xs)[0]
            meta_ps.append(mp)

        P_final = np.mean(np.vstack(meta_ps), axis=0)
        cls_idx = int(np.argmax(P_final))
        label   = int(le.inverse_transform([cls_idx])[0])
        ids.append(p.stem); ranks.append(label)

    sub = pd.DataFrame({'id': ids, 'rank': ranks})
    outp = Path(args.out_csv); outp.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(outp, index=False, encoding='utf-8-sig')
    print(f"submission saved -> {outp} (rows={len(sub)})")

if __name__ == '__main__':
    main()
