"""
slug_core.py - everything the dashboard needs that is NOT the user interface.

Keeping the logic here (and not inside app.py) means it can be tested without
Streamlit, and the numbers on the dashboard are the same numbers as in the
Colab notebook.

Scope reminder (from the project report):
  * the classes are a PROXY flow-stability label, not OLGA flow regimes;
  * the models IDENTIFY the state of the current record, they do not forecast;
  * the recommendation rules are prompts for an engineer, not instructions.
"""
from __future__ import annotations

import itertools
import math

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Constants (identical to the final 7-input model of the report)
# ----------------------------------------------------------------------------
FEATURES = ['QLT', 'UL', 'UG', 'VALVDP', 'gas_lift_MMSCFD',
            'topside_choke_TCV_pct_open', 'is_choked']
REQUIRED_SIGNALS = ['QLT', 'UL', 'UG']

FEATURE_INFO = {
    'QLT': ('Total liquid flow at the riser outlet', 'm³/s'),
    'UL': ('Liquid film velocity at the riser outlet', 'm/s'),
    'UG': ('Gas velocity at the riser outlet', 'm/s'),
    'VALVDP': ('Pressure drop across the topside choke', 'bar'),
    'gas_lift_MMSCFD': ('Gas-lift rate', 'MMSCFD'),
    'topside_choke_TCV_pct_open': ('Topside choke opening', '% open'),
    'is_choked': ('Topside choke restricted (1) or not (0)', 'flag'),
}

CLASS_SEVERE = 'Severe-Slugging Signature'
CLASS_STABLE = 'Stable / Continuous Flow'
CLASS_TRANS = 'Transitional / Mildly Oscillatory'
CLASSES = [CLASS_SEVERE, CLASS_STABLE, CLASS_TRANS]          # scikit-learn's (alphabetical) order
SHORT = {CLASS_SEVERE: 'Severe', CLASS_STABLE: 'Stable', CLASS_TRANS: 'Transitional'}

STARTUP_HOURS = 4.1      # first hours of a run: labels may reflect start-up (report, section 3.1)
RECENT_N = 9             # about 2 h at 15-minute sampling
PROXY_WINDOW = 9
AMP_SEVERE, AMP_TRANS, BACKFLOW_SEVERE = 1.0, 0.3, 0.3


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
def load_training_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values(['scenario_id', 'TIME_h']).reset_index(drop=True)


def describe_scenario(g: pd.DataFrame) -> str:
    gl = float(g['gas_lift_MMSCFD'].iloc[0])
    tcv = g['topside_choke_TCV_pct_open'].iloc[0]
    well = g['well_branch'].iloc[0] if 'well_branch' in g.columns else '?'
    choke = 'choke unrestricted' if pd.isna(tcv) else f'choke {tcv:.0f}% open'
    return f"{g['scenario_id'].iloc[0]} · {well} · {gl:.0f} MMSCFD gas lift · {choke}"


def fill_settings(df: pd.DataFrame, gas_lift=None, tcv_open=None) -> pd.DataFrame:
    """Make sure every model input exists. Settings that are already columns
    (a live stream may change them) are kept; missing ones come from the arguments."""
    out = df.copy()
    if 'gas_lift_MMSCFD' not in out.columns:
        out['gas_lift_MMSCFD'] = np.nan if gas_lift is None else float(gas_lift)
    if 'topside_choke_TCV_pct_open' not in out.columns:
        out['topside_choke_TCV_pct_open'] = np.nan if tcv_open is None else float(tcv_open)
    if 'is_choked' not in out.columns:
        out['is_choked'] = out['topside_choke_TCV_pct_open'].notna().astype(int)
    for c in FEATURES:
        if c not in out.columns:
            out[c] = np.nan
    return out


def proxy_label(df: pd.DataFrame, window: int = PROXY_WINDOW) -> pd.Series:
    """The transparent proxy label of the report (equations 3 to 5). REFERENCE ONLY."""
    q = df['QLT'].reset_index(drop=True)
    ul = df['UL'].reset_index(drop=True)
    amp = (q.rolling(window, center=True, min_periods=1).max()
           - q.rolling(window, center=True, min_periods=1).min()) / (q.abs().mean() + 1e-9)
    backflow = (ul < 0).astype(int).rolling(window, center=True, min_periods=1).mean()
    label = np.where((amp > AMP_SEVERE) | (backflow > BACKFLOW_SEVERE), CLASS_SEVERE,
                     np.where((amp > AMP_TRANS) | (backflow > 0), CLASS_TRANS, CLASS_STABLE))
    return pd.Series(label, index=df.index)


# ----------------------------------------------------------------------------
# Models (same settings as the Colab notebook)
# ----------------------------------------------------------------------------
def make_models() -> dict:
    return {
        'HistGB': HistGradientBoostingClassifier(random_state=42, max_iter=200),
        'LogReg': Pipeline([
            ('imp', SimpleImputer(strategy='median', add_indicator=True)),
            ('sc', StandardScaler()),
            ('lr', LogisticRegression(max_iter=2000, class_weight='balanced')),
        ]),
    }


def fit_models(train_df: pd.DataFrame) -> dict:
    X, y = train_df[FEATURES], train_df['proxy_label']
    return {name: clone(m).fit(X, y) for name, m in make_models().items()}


def make_background(train_df: pd.DataFrame, n: int = 100, seed: int = 42) -> pd.DataFrame:
    X = train_df[FEATURES]
    return X.sample(n=min(n, len(X)), random_state=seed).reset_index(drop=True)


def save_bundle(path: str, models: dict, background: pd.DataFrame) -> None:
    joblib.dump({'models': models, 'background': background, 'features': FEATURES,
                 'classes': CLASSES, 'sklearn_version': sklearn.__version__}, path)


def load_bundle(path: str):
    """(models, background) or None if the file is missing, unreadable or was
    saved with a different scikit-learn version (pickles are version-sensitive)."""
    try:
        b = joblib.load(path)
        if b.get('sklearn_version') != sklearn.__version__ or b.get('features') != FEATURES:
            return None
        return b['models'], b['background']
    except Exception:
        return None


def predict_all(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES]
    out = pd.DataFrame(index=df.index)
    for name, m in models.items():
        P = m.predict_proba(X)
        classes = list(m.classes_)
        for k, c in enumerate(classes):
            out[f'{name}_p_{SHORT[c]}'] = P[:, k]
        out[f'{name}_class'] = [classes[i] for i in P.argmax(axis=1)]
        out[f'{name}_conf'] = P.max(axis=1)
    out['agree'] = out['HistGB_class'] == out['LogReg_class']
    return out


# ----------------------------------------------------------------------------
# Data quality
# ----------------------------------------------------------------------------
def reference_stats(train_df: pd.DataFrame) -> dict:
    stats = {}
    for c in FEATURES:
        s = train_df[c].dropna()
        stats[c] = {'min': float(s.min()), 'max': float(s.max())} if len(s) else None
    return stats


def reference_median_max_abs(train_df: pd.DataFrame) -> dict:
    return {c: float(train_df.groupby('scenario_id')[c].apply(lambda s: s.abs().max()).median())
            for c in REQUIRED_SIGNALS}


def quality_flags(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """One row per record: status ('ok' / 'warning' / 'insufficient') and a note."""
    n = len(df)
    notes = [[] for _ in range(n)]
    status = np.array(['ok'] * n, dtype=object)

    missing = df[REQUIRED_SIGNALS].isna().any(axis=1).to_numpy()
    for i in np.where(missing)[0]:
        notes[i].append('QLT, UL or UG missing')
    status[missing] = 'insufficient'

    for c in FEATURES:
        st_ = stats.get(c)
        if st_ is None or c not in df.columns:
            continue
        lo, hi = st_['min'], st_['max']
        margin = 0.1 * (hi - lo) if hi > lo else 0.0
        v = df[c].to_numpy(dtype=float)
        bad = (v < lo - margin) | (v > hi + margin)         # NaN compares False
        for i in np.where(bad)[0]:
            notes[i].append(f'{c} = {v[i]:.4g} outside training range [{lo:.4g}, {hi:.4g}]')
        status[bad & (status == 'ok')] = 'warning'

    if 'TIME_h' in df.columns and n > 2:
        t = df['TIME_h'].to_numpy(dtype=float)
        dt = np.diff(t, prepend=np.nan)
        med = np.nanmedian(dt[1:]) if n > 2 else np.nan
        gap = np.zeros(n, bool)
        if np.isfinite(med) and med > 0:
            gap = (dt > 2.5 * med) | (dt <= 0)
        for i in np.where(gap)[0]:
            notes[i].append('time gap or repeated/out-of-order timestamp')
        status[gap & (status == 'ok')] = 'warning'

    return pd.DataFrame({'status': status, 'note': ['; '.join(x) for x in notes]}, index=df.index)


def scenario_quality_messages(df: pd.DataFrame) -> list:
    msgs = []
    if 'QLT' in df and df['QLT'].notna().any() and df['QLT'].abs().median() > 1.0:
        msgs.append('QLT values are larger than 1 - are they in m³/h? The model expects m³/s.')
    if 'VALVDP' not in df or df['VALVDP'].isna().all():
        msgs.append('VALVDP not supplied. The model treats a missing VALVDP like the OP2 cases '
                    '(report, section 3.4), so read the flag with extra caution.')
    for c in REQUIRED_SIGNALS:
        if c not in df or df[c].isna().all():
            msgs.append(f'{c} is missing for the whole record - predictions are not meaningful.')
    return msgs


# ----------------------------------------------------------------------------
# Explanation: exact Shapley values (the quantity SHAP's permutation explainer estimates)
# ----------------------------------------------------------------------------
def shapley_exact(model, x_row: pd.DataFrame, background: pd.DataFrame):
    """Exact interventional Shapley values over all 2^M feature coalitions (M = 7, so 128).
    value(S) = mean model output when the features in S take this record's values and the
    others are drawn from the background rows.  Returns phi (M x K), base (K), fx (K), where
    base + phi.sum(0) == fx exactly (the additivity check of report equation 9)."""
    cols = FEATURES
    M = len(cols)
    xv = np.asarray(x_row[cols], dtype=float).reshape(M)
    B = np.asarray(background[cols], dtype=float)
    nb = B.shape[0]
    masks = np.array(list(itertools.product([False, True], repeat=M)))        # (2^M, M)
    data = np.where(masks[:, None, :], xv[None, None, :], B[None, :, :]).reshape(-1, M)
    P = model.predict_proba(pd.DataFrame(data, columns=cols))
    K = P.shape[1]
    V = P.reshape(2 ** M, nb, K).mean(axis=1)                                  # (2^M, K)

    pow2 = 2 ** np.arange(M - 1, -1, -1)
    size = masks.sum(axis=1)
    w = np.array([math.factorial(s) * math.factorial(M - s - 1) / math.factorial(M) for s in range(M)])
    idx = masks.astype(int) @ pow2
    phi = np.zeros((M, K))
    for i in range(M):
        without = ~masks[:, i]
        s_idx = idx[without]
        phi[i] = (w[size[without]][:, None] * (V[s_idx + pow2[i]] - V[s_idx])).sum(axis=0)
    return phi, V[0], V[-1]


def explain_record(model, x_row: pd.DataFrame, background: pd.DataFrame) -> dict:
    phi, base, fx = shapley_exact(model, x_row, background)
    table = pd.DataFrame({
        'feature': FEATURES,
        'meaning': [FEATURE_INFO[c][0] for c in FEATURES],
        'unit': [FEATURE_INFO[c][1] for c in FEATURES],
        'value': [x_row[c].iloc[0] for c in FEATURES],
    })
    for k, c in enumerate(model.classes_):
        table[f'phi_{SHORT[c]}'] = phi[:, k]
    return {'table': table, 'base': dict(zip([SHORT[c] for c in model.classes_], base)),
            'fx': dict(zip([SHORT[c] for c in model.classes_], fx)),
            'additivity_error': float(np.abs(base + phi.sum(axis=0) - fx).max())}


def global_importance(model, X_sample: pd.DataFrame, background: pd.DataFrame, class_name: str) -> pd.Series:
    k = list(model.classes_).index(class_name)
    acc = np.zeros(len(FEATURES))
    for i in range(len(X_sample)):
        phi, _, _ = shapley_exact(model, X_sample.iloc[[i]], background)
        acc += np.abs(phi[:, k])
    return pd.Series(acc / max(len(X_sample), 1), index=FEATURES).sort_values(ascending=False)


# ----------------------------------------------------------------------------
# Priority and engineering rules (prompts for an engineer, not instructions)
# ----------------------------------------------------------------------------
GREEN, AMBER, RED, GREY = '#30a46c', '#f5a524', '#e5484d', '#8b98a9'


def priority_for(hgb_class: str, lr_class: str, quality_status: str):
    if quality_status == 'insufficient':
        return 'Data insufficient', GREY
    if hgb_class != lr_class:
        return 'Models disagree: engineer review', AMBER
    return {CLASS_STABLE: ('Monitor', GREEN),
            CLASS_TRANS: ('Watch', AMBER),
            CLASS_SEVERE: ('High-priority review', RED)}[hgb_class]


def recommend(ctx: dict) -> dict:
    """ctx keys: hgb_class, lr_class, agree, quality_status, quality_note, gas_lift, tcv_open
    (None = unrestricted), valvdp_available, time_h, recent_severe_share, drivers
    (list of (feature, phi_for_predicted_class)).  Evidence figures come from the nine
    simulated scenarios of the report (proxy label) and are NOT validated for operations."""
    priority, colour = priority_for(ctx['hgb_class'], ctx['lr_class'], ctx['quality_status'])
    out = {'priority': priority, 'colour': colour, 'checks': [], 'options': [], 'notes': []}

    if ctx['quality_status'] == 'insufficient':
        out['headline'] = 'The inputs are not good enough to give a recommendation.'
        out['checks'] = ['Restore or verify the QLT, UL and UG feed', 'Check units, timestamps and instruments']
        return out

    if ctx['quality_status'] == 'warning':
        out['notes'].append('Input warning: ' + (ctx.get('quality_note') or 'see data-quality panel') +
                            '. The model is being used outside the conditions it was trained on.')

    if not ctx['agree']:
        out['headline'] = 'The two models give different answers, so the flag is not reliable.'
        out['checks'] = ['Judge the trends yourself', 'Compare both explanations', 'Check inputs and settings']
        return out

    cls = ctx['hgb_class']
    gl, tcv = ctx['gas_lift'], ctx['tcv_open']
    out['headline'] = {CLASS_STABLE: 'Flow looks steady. Keep monitoring.',
                       CLASS_TRANS: 'Flow is unsteady. Watch it and review the evidence.',
                       CLASS_SEVERE: 'The record carries the severe-slugging signature. Review now.'}[cls]

    if cls == CLASS_STABLE:
        out['checks'] = ['Keep monitoring']
    else:
        out['checks'] = ['Verify the trend evidence', 'Check separator and compressor margin',
                         'Review the approved slug procedure']
        if cls == CLASS_SEVERE:
            out['checks'].append('Escalate to the responsible engineer')

        top = [f for f, _ in (ctx.get('drivers') or [])[:3]]
        if gl is not None and not pd.isna(gl) and gl <= 5:
            agree_txt = ' The explanation also lists gas lift as a main driver.' if 'gas_lift_MMSCFD' in top else ''
            out['options'].append({
                'action': 'Raise gas lift, within compressor and riser limits',
                'evidence': 'Simulated scenarios: Severe share was 100% at 0 MMSCFD, 88.9-94.1% at 5 MMSCFD '
                            'and 1.7% at 10-15 MMSCFD.' + agree_txt,
                'caveat': 'Gas lift took only four values and was fixed inside each scenario.'})
        if gl is not None and not pd.isna(gl) and gl > 0 and tcv is None:
            out['options'].append({
                'action': 'Close the topside choke in steps, watching VALVDP and pressure limits',
                'evidence': 'OP1 at 5 MMSCFD: 88.9% Severe unchoked, 3.1% at 10% open, 89.3% Stable at 5% open.',
                'caveat': 'Each choke setting rests on one scenario. Stay inside separator and pressure limits.'})
        if gl is not None and not pd.isna(gl) and gl == 0:
            out['options'].append({
                'action': 'Do not rely on choking alone',
                'evidence': 'The OP2 case with a 10% open choke and no gas lift stayed 100% Severe.',
                'caveat': 'One case only, and it is a different well from the OP1 choke cases.'})

        if ctx.get('recent_severe_share') is not None and ctx['recent_severe_share'] >= 0.5:
            out['notes'].append(f"{ctx['recent_severe_share']:.0%} of the last {RECENT_N} records were flagged severe, "
                                'so this is persistent, not a single noisy record.')
        if ctx['time_h'] is not None and ctx['time_h'] < STARTUP_HOURS:
            out['options'].insert(0, {
                'action': 'Wait and re-check before acting',
                'evidence': 'In the simulations, the severe labels of the stable scenarios sit in the first 1-3.75 h.',
                'caveat': 'It is not known whether these hours are numerical start-up or a real response.'})

    if not ctx['valvdp_available']:
        out['notes'].append('VALVDP is missing, and the model treats that like the OP2 cases. '
                            'Read this flag with extra caution.')
    return out


# ----------------------------------------------------------------------------
# Leave-one-scenario-out check (reproduces the 5-of-9 / 7-of-9 result of the report)
# ----------------------------------------------------------------------------
def loso_table(train_df: pd.DataFrame, min_time: float = 0.0, progress=None) -> pd.DataFrame:
    d = train_df[train_df['TIME_h'] >= min_time]
    scen = list(d['scenario_id'].unique())
    rows = []
    for i, held in enumerate(scen):
        tr, te = d[d['scenario_id'] != held], d[d['scenario_id'] == held]
        models = fit_models(tr)
        row = {'scenario': held, 'dominant_proxy_class': te['proxy_label'].value_counts().index[0]}
        for name, m in models.items():
            p = pd.Series(m.predict(te[FEATURES]))
            row[name] = p.value_counts().index[0]
            row[f'{name}_rows_right_%'] = round(float((p.values == te['proxy_label'].values).mean() * 100), 1)
        rows.append(row)
        if progress:
            progress((i + 1) / len(scen))
    return pd.DataFrame(rows)
