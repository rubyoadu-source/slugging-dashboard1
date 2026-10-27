"""
SlugGuard - explainable decision support for slugging risk (research prototype).

What it does : classifies the CURRENT record into a proxy flow-stability class with two
               models, explains the flag (exact Shapley values), and lists engineering
               checks and options for an engineer to review.
What it is NOT: a forecast, a validated slugging detector, or a control system.
Run locally  : streamlit run app.py
"""
import html
import io
import os
import tempfile
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import slug_core as sc
from consolidate import consolidate_scenarios
from ingest import load_workbook_scenarios

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(HERE, 'artifacts', 'models_full.joblib')

st.set_page_config(page_title='SlugGuard | Flow assurance decision support',
                   page_icon='🌊', layout='wide')


def list_repo_files(limit=80):
    out = []
    for root, dirs, files in os.walk(HERE):
        dirs[:] = sorted(d for d in dirs if d not in ('__pycache__', '.git'))
        for f in sorted(files):
            out.append(os.path.relpath(os.path.join(root, f), HERE))
    return out[:limit]


def find_training_csv():
    """data/training_data.csv if it is there; otherwise any file that looks like it
    (Linux paths are case-sensitive and the GitHub web uploader can flatten folders)."""
    exact = os.path.join(HERE, 'data', 'training_data.csv')
    if os.path.exists(exact):
        return exact
    hits = []
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git')]
        for f in files:
            name = f.lower().replace(' ', '')
            if name.startswith('training') and name.endswith('.csv'):
                hits.append(os.path.join(root, f))
    return sorted(hits, key=len)[0] if hits else None


DATA_PATH = find_training_csv()
if DATA_PATH is None or os.path.getsize(DATA_PATH) < 1000:
    st.error('The training data file was not found (or is empty): expected `data/training_data.csv` '
             'in the repository, next to `app.py`.')
    st.markdown('Files the app can see in the repository:')
    st.code('\n'.join(list_repo_files()))
    st.markdown('Check that the folder is called exactly `data` (lower case) and holds `training_data.csv`, '
                'then reboot the app from the Streamlit menu.')
    st.stop()

CLASS_COL = {sc.CLASS_SEVERE: '#e5484d', sc.CLASS_STABLE: '#30a46c', sc.CLASS_TRANS: '#f5a524'}
CLASS_RGBA = {sc.CLASS_SEVERE: 'rgba(229,72,77,0.55)', sc.CLASS_STABLE: 'rgba(48,164,108,0.55)',
              sc.CLASS_TRANS: 'rgba(245,165,36,0.55)'}
TEAL = '#2dd4bf'

st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 1.2rem; max-width: 1500px;}
.sg-top {display:flex; justify-content:space-between; align-items:center; margin-bottom:.7rem;}
.sg-title {font-size:1.3rem; font-weight:700; letter-spacing:.01em;}
.sg-title span {color:#7f90a8; font-weight:400; font-size:.9rem; margin-left:.5rem;}
.sg-pill {display:inline-block; border:1px solid #2a3b57; color:#9fb0c7; border-radius:999px;
          padding:.15rem .65rem; font-size:.68rem; letter-spacing:.08em; margin-left:.4rem; cursor:help;}
.sg-h {color:#7f90a8; font-size:.68rem; text-transform:uppercase; letter-spacing:.1em; margin-bottom:.35rem;}
.sg-hero {border-radius:16px; padding:1rem 1.3rem; min-height:150px;}
.sg-state {font-size:1.75rem; font-weight:800; line-height:1.1;}
.sg-sub2 {color:#d3dcea; font-size:1rem; margin-top:.3rem;}
.sg-meta {color:#7f90a8; font-size:.78rem; margin-top:.9rem;}
.sg-chip {display:inline-block; background:#121e32; border:1px solid #223350; color:#c6d2e4;
          border-radius:999px; padding:.18rem .65rem; font-size:.78rem; margin:0 .35rem .35rem 0; cursor:default;}
.sg-dot {display:inline-block; width:.5rem; height:.5rem; border-radius:50%; margin-right:.4rem;}
.sg-card {background:#0f1a2c; border:1px solid #1c2c46; border-radius:16px; padding:.9rem 1.1rem;}
.sg-list {margin:.2rem 0 .5rem 0; padding-left:1.1rem; color:#c6d2e4; font-size:.88rem;}
.sg-opt {border-top:1px solid #1c2c46; padding:.45rem 0; font-size:.9rem;}
.sg-opt details {color:#8ea0b8; font-size:.8rem; margin-top:.15rem;}
.sg-opt summary {cursor:pointer;}
.sg-foot {color:#6f819a; font-size:.72rem; margin-top:.4rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- cached resources
@st.cache_data(show_spinner=False)
def load_train():
    return sc.load_training_data(DATA_PATH)


TRAIN = load_train()
STATS = sc.reference_stats(TRAIN)
SCEN_DESC = {sid: sc.describe_scenario(g) for sid, g in TRAIN.groupby('scenario_id')}


@st.cache_resource(show_spinner='Preparing the models (first load only)...')
def get_models(held_out):
    """held_out=None: all scenarios; otherwise that scenario is left out of training."""
    if held_out is None:
        loaded = sc.load_bundle(BUNDLE_PATH)
        if loaded is not None:
            return loaded[0], loaded[1], 'Colab export (artifacts/models_full.joblib)'
        d = TRAIN
    else:
        d = TRAIN[TRAIN['scenario_id'] != held_out]
    return sc.fit_models(d), sc.make_background(d), 'trained in the app from data/training_data.csv'


@st.cache_data(show_spinner=False)
def run_predictions(X, held_out):
    models, _, _ = get_models(held_out)
    return sc.predict_all(models, X)


@st.cache_data(show_spinner=False)
def explain_cached(row, held_out, model_name):
    models, bg, _ = get_models(held_out)
    return sc.explain_record(models[model_name], row, bg)


@st.cache_data(show_spinner=False)
def global_cached(sample, held_out, class_name):
    models, bg, _ = get_models(held_out)
    return sc.global_importance(models['HistGB'], sample, bg.sample(n=min(30, len(bg)), random_state=42), class_name)


@st.cache_data(show_spinner=False)
def read_workbooks(items):
    raw = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, content in items:
            path = os.path.join(tmp, os.path.basename(name))
            with open(path, 'wb') as f:
                f.write(content)
            raw += load_workbook_scenarios(path)
    return consolidate_scenarios(raw, reference_median_max_abs=sc.reference_median_max_abs(TRAIN))


@st.cache_data(show_spinner=False)
def loso_cached(min_time):
    return sc.loso_table(TRAIN, min_time=min_time)


# ---------------------------------------------------------------- small helpers
def normalise_columns(df):
    mapping = {}
    lower = {t.lower(): t for t in sc.FEATURES}
    for c in df.columns:
        k = str(c).strip().lower().replace(' ', '_')
        if k in lower:
            mapping[c] = lower[k]
        elif k in ('time', 'time_h', 'time_(h)', 'time1', 't_h'):
            mapping[c] = 'TIME_h'
    out = df.rename(columns=mapping)
    return out.loc[:, ~out.columns.duplicated()]


def settings_widgets(gas_lift_known, choke_known):
    gl, tcv = None, None
    if not gas_lift_known:
        gl = st.sidebar.number_input('Gas-lift rate (MMSCFD)', 0.0, 30.0, 5.0, 1.0)
    if not choke_known:
        restricted = st.sidebar.radio('Topside choke', ['Unrestricted', 'Restricted'], horizontal=True)
        if restricted == 'Restricted':
            tcv = st.sidebar.number_input('Choke opening (% open)', 1.0, 100.0, 10.0, 1.0)
    return gl, tcv


# ---------------------------------------------------------------- sidebar: data source
source = st.sidebar.radio('Data', ['Demo scenarios', 'Upload CSV', 'Upload OLGA workbook'])

held_out, df, label, notes = None, None, '', []

if source == 'Demo scenarios':
    sid = st.sidebar.selectbox('Scenario', list(SCEN_DESC.keys()), format_func=lambda s: SCEN_DESC[s])
    honest = st.sidebar.toggle('Honest view', value=True,
                               help='On: the models were trained WITHOUT this scenario. '
                                    'Off: trained on all nine, so they look better than they will on a new case.')
    held_out = sid if honest else None
    df = TRAIN[TRAIN['scenario_id'] == sid].sort_values('TIME_h').reset_index(drop=True)
    label = SCEN_DESC[sid]

elif source == 'Upload CSV':
    up = st.sidebar.file_uploader('CSV with QLT, UL, UG (optional: TIME_h, VALVDP)', type=['csv'])
    if up is None:
        st.info('Upload a CSV with columns QLT (m³/s), UL (m/s), UG (m/s). TIME_h (hours) and VALVDP (bar) '
                'are optional. Gas-lift rate and choke opening can be columns or sidebar entries.')
        st.stop()
    raw_df = normalise_columns(pd.read_csv(io.BytesIO(up.getvalue())))
    if not all(c in raw_df.columns for c in sc.REQUIRED_SIGNALS):
        st.error('The CSV needs QLT, UL and UG columns.')
        st.stop()
    if 'TIME_h' not in raw_df.columns:
        raw_df['TIME_h'] = np.arange(len(raw_df)) * 0.25
        notes.append('No time column found: assumed 15-minute steps.')
    gl, tcv = settings_widgets('gas_lift_MMSCFD' in raw_df.columns,
                               'topside_choke_TCV_pct_open' in raw_df.columns)
    df = sc.fill_settings(raw_df.sort_values('TIME_h').reset_index(drop=True), gl, tcv)
    label = f'Uploaded CSV: {up.name}'

else:
    ups = st.sidebar.file_uploader('OLGA workbook(s) (.xlsx)', type=['xlsx'], accept_multiple_files=True)
    if not ups:
        st.info('Upload one or more OLGA export workbooks. The reader rebuilds the values from the Excel '
                'formulas, runs the corruption and duplicate checks, and takes gas lift and choke from the case name.')
        st.stop()
    meta, master, qlog = read_workbooks(tuple((u.name, u.getvalue()) for u in ups))
    if len(meta) == 0:
        st.error('No valid scenario found. See the quality log below.')
        st.dataframe(qlog, hide_index=True)
        st.stop()
    with st.sidebar.expander('Quality log of the upload'):
        st.dataframe(qlog[['source_sheet', 'finding', 'action']] if len(qlog) else qlog, hide_index=True)
    ids = list(meta['scenario_id'])
    fmt = {r.scenario_id: f'{r.scenario_id} · {r.well_branch} · {r.gas_lift_condition}'
           + ('' if pd.isna(r.topside_choke_TCV_pct_open) else f' · choke {r.topside_choke_TCV_pct_open:.0f}% open')
           for r in meta.itertuples()}
    sid = st.sidebar.selectbox('Scenario in upload', ids, format_func=lambda s: fmt[s])
    m = meta[meta['scenario_id'] == sid].iloc[0]
    d = master[master['scenario_id'] == sid].sort_values('TIME_h').reset_index(drop=True)
    parsed = m['parse_warning'] is None or pd.isna(m['parse_warning'])
    gl_meta = m['gas_lift_MMSCFD'] if parsed else None
    tcv_meta = None if (not parsed or pd.isna(m['topside_choke_TCV_pct_open'])) else float(m['topside_choke_TCV_pct_open'])
    gl, tcv = settings_widgets(parsed, parsed)
    df = sc.fill_settings(d[[c for c in ['TIME_h', 'QLT', 'UL', 'UG', 'VALVDP'] if c in d.columns]],
                          gl_meta if parsed else gl, tcv_meta if parsed else tcv)
    label = fmt[sid]
    if parsed:
        notes.append('Gas lift and choke were read from the case name.')

if not all(c in df.columns for c in sc.REQUIRED_SIGNALS) or len(df) < 3:
    st.error('Not enough data to show a record.')
    st.stop()

df['proxy_label'] = sc.proxy_label(df)
n = len(df)

# new dataset -> reset the replay
data_id = f'{source}|{label}|{n}|{held_out}|{float(df["QLT"].sum()):.6g}'
if st.session_state.get('data_id') != data_id:
    st.session_state['data_id'] = data_id
    st.session_state['pos'] = 0
    st.session_state['playing'] = False
st.session_state['n'] = n
st.session_state.setdefault('playing', False)
if st.session_state.pop('_advance', False):
    st.session_state['pos'] = min(st.session_state.get('pos', 0) + 1, n - 1)
st.session_state['pos'] = int(min(max(st.session_state.get('pos', 0), 0), n - 1))


def _toggle():
    st.session_state['playing'] = not st.session_state['playing']


def _reset():
    st.session_state['pos'] = 0
    st.session_state['playing'] = False


def _step():
    st.session_state['pos'] = min(st.session_state['pos'] + 1, st.session_state['n'] - 1)


# ---------------------------------------------------------------- sidebar: replay (compact)
st.sidebar.markdown('---')
b1, b2, b3 = st.sidebar.columns(3)
b1.button('⏸' if st.session_state['playing'] else '▶', on_click=_toggle, key='btn_play',
          help='Play / pause the simulated live stream')
b2.button('⏮', on_click=_reset, key='btn_reset', help='Back to the first record')
b3.button('⏭', on_click=_step, key='btn_step', help='Next record')
st.sidebar.slider('Record', 0, n - 1, key='pos')
speed_name = st.sidebar.select_slider('Speed', options=['Slow', 'Normal', 'Fast'], value='Normal')
speed = {'Slow': 0.5, 'Normal': 1.0, 'Fast': 2.0}[speed_name]
signal_options = ['QLT', 'UL', 'UG'] + (['VALVDP'] if df['VALVDP'].notna().any() else [])
signals = st.sidebar.multiselect('Signals', signal_options, default=['QLT'])
if not signals:
    signals = ['QLT']
pos = st.session_state['pos']
show_proxy = source == 'Demo scenarios'

# ---------------------------------------------------------------- predictions
models, background, model_source = get_models(held_out)
preds = run_predictions(df[sc.FEATURES], held_out)
qual = sc.quality_flags(df, STATS)
row, pr, q = df.iloc[pos], preds.iloc[pos], qual.iloc[pos]
hgb_cls, lr_cls = pr['HistGB_class'], pr['LogReg_class']
priority, p_col = sc.priority_for(hgb_cls, lr_cls, q['status'])
recent = preds['HistGB_class'].iloc[max(0, pos - sc.RECENT_N + 1):pos + 1]
recent_share = float((recent == sc.CLASS_SEVERE).mean())
exp = explain_cached(df.iloc[[pos]][sc.FEATURES], held_out, 'HistGB')
exp_col = f'phi_{sc.SHORT[hgb_cls]}'
pushes = exp['table'][exp['table'][exp_col] > 0].sort_values(exp_col, ascending=False)
drivers = list(zip(pushes['feature'], pushes[exp_col]))
tcv_val = None if pd.isna(row['topside_choke_TCV_pct_open']) else float(row['topside_choke_TCV_pct_open'])
gl_val = None if pd.isna(row['gas_lift_MMSCFD']) else float(row['gas_lift_MMSCFD'])
rec = sc.recommend(dict(
    hgb_class=hgb_cls, lr_class=lr_cls, agree=bool(pr['agree']), quality_status=q['status'],
    quality_note=q['note'], gas_lift=gl_val, tcv_open=tcv_val, valvdp_available=not pd.isna(row['VALVDP']),
    time_h=float(row['TIME_h']), recent_severe_share=recent_share, drivers=drivers))


# ---------------------------------------------------------------- small html helpers
def esc(s):
    return html.escape(str(s), quote=True)


def chip(text, colour=None, tip=''):
    dot = f'<span class="sg-dot" style="background:{colour}"></span>' if colour else ''
    return f'<span class="sg-chip" title="{esc(tip)}">{dot}{esc(text)}</span>'


HERO = {'Monitor': ('MONITOR', 'Flow looks steady'),
        'Watch': ('WATCH', 'Flow is unsteady'),
        'High-priority review': ('HIGH-PRIORITY REVIEW', 'Severe-slugging signature'),
        'Models disagree: engineer review': ('MODELS DISAGREE', 'Engineer review needed'),
        'Data insufficient': ('DATA INSUFFICIENT', 'Inputs are not reliable')}

# ---------------------------------------------------------------- header
if source == 'Demo scenarios':
    pill1 = ('SIMULATION', 'Simulated OLGA scenarios, not field data.')
else:
    pill1 = ('UPLOADED DATA · NOT VALIDATED',
             'The models were trained on nine simulated scenarios of one design case. '
             'Their scores for any other system are not validated.')
st.markdown(
    '<div class="sg-top"><div class="sg-title">🌊 SlugGuard<span>flow-assurance decision support</span></div><div>'
    f'<span class="sg-pill" title="{esc(pill1[1])}">{esc(pill1[0])}</span>'
    '<span class="sg-pill" title="Each record is classified on its own. Nothing here forecasts a future event.">'
    'IDENTIFIES NOW · NO FORECAST</span>'
    '<span class="sg-pill" title="The dashboard never sends commands to equipment.">READ-ONLY</span>'
    '</div></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- hero row
state, subtitle = HERO[priority]
meta = f'{label}  ·  record {pos + 1}/{n}  ·  t = {row["TIME_h"]:.2f} h' + ('  ·  ▶ replay' if st.session_state['playing'] else '')
hero_col, prob_col = st.columns([2, 3])
hero_col.markdown(
    f'<div class="sg-hero" style="background:linear-gradient(135deg,{p_col}2e,#0e1828 78%);border:1px solid {p_col}88">'
    '<div class="sg-h">Current assessment</div>'
    f'<div class="sg-state" style="color:{p_col}">{esc(state)}</div>'
    f'<div class="sg-sub2">{esc(subtitle)}</div>'
    f'<div class="sg-meta">{esc(meta)}</div></div>', unsafe_allow_html=True)


def prob_figure():
    names = ['Gradient boosting', 'Logistic regression']
    fig = go.Figure()
    for cls in (sc.CLASS_STABLE, sc.CLASS_TRANS, sc.CLASS_SEVERE):
        vals = [float(pr[f'HistGB_p_{sc.SHORT[cls]}']), float(pr[f'LogReg_p_{sc.SHORT[cls]}'])]
        fig.add_trace(go.Bar(y=names, x=vals, orientation='h', name=sc.SHORT[cls], marker_color=CLASS_COL[cls],
                             text=[f'{v:.0%}' if v >= 0.07 else '' for v in vals], textposition='inside',
                             insidetextanchor='middle',
                             hovertemplate='%{y}: %{x:.2f}<extra>' + sc.SHORT[cls] + '</extra>'))
    fig.update_layout(template='plotly_dark', barmode='stack', height=118, margin=dict(l=0, r=0, t=24, b=0),
                      xaxis=dict(range=[0, 1], visible=False), yaxis=dict(autorange='reversed'),
                      legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='left', x=0),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


with prob_col:
    st.markdown('<div class="sg-h">Class scores · uncalibrated</div>', unsafe_allow_html=True)
    st.plotly_chart(prob_figure())
    q_col = {'ok': sc.GREEN, 'warning': sc.AMBER, 'insufficient': sc.GREY}[q['status']]
    q_txt = {'ok': 'Data OK', 'warning': 'Data warning', 'insufficient': 'Data insufficient'}[q['status']]
    chips = [chip('Models agree' if pr['agree'] else 'Models disagree', sc.GREEN if pr['agree'] else sc.AMBER,
                  'Agreement is not proof of correctness.'),
             chip(q_txt, q_col, q['note'] or 'All inputs are inside the training range.'),
             chip(f'Severe flags {recent_share:.0%} · last 2 h', sc.RED if recent_share >= 0.5 else None,
                  f'Share of the last {len(recent)} records flagged severe (looks backwards, not forwards).'),
             chip('Gas lift ' + ('?' if gl_val is None else f'{gl_val:g} MMSCFD')),
             chip('Choke unrestricted' if tcv_val is None else f'Choke {tcv_val:g}% open')]
    if source == 'Demo scenarios':
        chips.append(chip('Honest view', TEAL, 'The models were trained without this scenario.') if held_out
                     else chip('In-sample', sc.AMBER, 'The models were trained on this scenario, so they look better '
                                                    'here than on a new case.'))
    info = sc.scenario_quality_messages(df) + notes + rec['notes']
    if info:
        chips.append(chip(f'ⓘ {len(info)} note' + ('s' if len(info) > 1 else ''), None, ' | '.join(info)))
    st.markdown(''.join(chips), unsafe_allow_html=True)


# ---------------------------------------------------------------- main chart
def trend_figure():
    d, p = df.iloc[:pos + 1], preds.iloc[:pos + 1]
    titles = {'QLT': 'QLT · m³/s', 'UL': 'UL · m/s', 'UG': 'UG · m/s', 'VALVDP': 'VALVDP · bar'}
    strips = [('HistGB_class', 'Gradient boosting'), ('LogReg_class', 'Logistic regression')]
    if show_proxy:
        strips.append(('proxy', 'Proxy label'))
    n_rows = len(signals) + len(strips)
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[1.0] * len(signals) + [0.14] * len(strips))
    marker_cols = [CLASS_COL[c] for c in p['HistGB_class']]
    for i, s in enumerate(signals, start=1):
        fig.add_trace(go.Scatter(x=d['TIME_h'], y=d[s], mode='lines+markers', showlegend=False,
                                 line=dict(color='rgba(45,212,191,0.55)', width=1.2),
                                 marker=dict(size=5, color=marker_cols)), row=i, col=1)
        fig.update_yaxes(title_text=titles[s], title_font=dict(size=11), row=i, col=1)
    step = float(np.nanmedian(np.diff(df['TIME_h']))) if n > 2 else 0.25
    for j, (col_name, title) in enumerate(strips, start=len(signals) + 1):
        series = d['proxy_label'] if col_name == 'proxy' else p[col_name]
        fig.add_trace(go.Bar(x=d['TIME_h'], y=[1] * len(d), width=step, showlegend=False,
                             marker_color=[CLASS_COL[c] for c in series],
                             hovertext=[sc.SHORT[c] for c in series], hoverinfo='text+x'), row=j, col=1)
        fig.update_yaxes(showticklabels=False, showgrid=False, range=[0, 1], title_text=title,
                         title_font=dict(size=9), row=j, col=1)
    fig.update_xaxes(range=[float(df['TIME_h'].min()), float(df['TIME_h'].max())])
    fig.update_xaxes(title_text='Time (h)', row=n_rows, col=1)
    fig.update_layout(template='plotly_dark', height=190 * len(signals) + 34 * len(strips) + 60, bargap=0,
                      margin=dict(l=0, r=0, t=6, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


st.markdown('<div class="sg-h" style="margin-top:.8rem">Trend · dots coloured by the gradient-boosting class</div>',
            unsafe_allow_html=True)
st.plotly_chart(trend_figure())

# ---------------------------------------------------------------- why + what to consider
why_col, rec_col = st.columns(2)


def why_figure():
    t = exp['table']
    t2 = t.assign(_abs=t[exp_col].abs()).sort_values('_abs', ascending=True)
    labels = [f'{r.feature} = ' + ('missing' if pd.isna(r.value) else f'{r.value:.3g}') for r in t2.itertuples()]
    fig = go.Figure(go.Bar(
        x=t2[exp_col], y=labels, orientation='h',
        marker_color=[CLASS_COL[sc.CLASS_SEVERE] if v > 0 else TEAL for v in t2[exp_col]],
        text=[f'{v:+.2f}' for v in t2[exp_col]], textposition='outside',
        customdata=list(t2['meaning']), hovertemplate='%{customdata}<br>contribution %{x:+.3f}<extra></extra>'))
    fig.update_layout(template='plotly_dark', height=330, margin=dict(l=0, r=30, t=6, b=0),
                      xaxis=dict(zeroline=True, zerolinecolor='#3a4d6e', showgrid=False, showticklabels=False),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


with why_col:
    st.markdown(f'<div class="sg-h">Why · what pushes the {esc(sc.SHORT[hgb_cls])} score '
                '(<span style="color:#e5484d">up</span> / <span style="color:#2dd4bf">down</span>)</div>',
                unsafe_allow_html=True)
    st.plotly_chart(why_figure())

with rec_col:
    body = '<div class="sg-h">What to consider</div>'
    if rec['checks']:
        body += '<ul class="sg-list">' + ''.join(f'<li>{esc(c)}</li>' for c in rec['checks']) + '</ul>'
    for o in rec['options']:
        body += (f'<div class="sg-opt"><b>{esc(o["action"])}</b><details><summary>Evidence and caveat</summary>'
                 f'<div>{esc(o["evidence"])}<br><i>{esc(o["caveat"])}</i></div></details></div>')
    body += '<div class="sg-foot">Prompts for engineer review, not instructions.</div>'
    st.markdown(f'<div class="sg-card">{body}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- details (only computed when switched on)
if st.toggle('Details', value=False, key='details_on'):
    t_explain, t_models, t_scen, t_log, t_about = st.tabs(['Explain more', 'Models', 'Scenario', 'Event log', 'About'])

    with t_explain:
        c1, c2 = st.columns(2)
        cls_choice = c1.selectbox('Class', sc.CLASSES, index=sc.CLASSES.index(hgb_cls),
                                  format_func=lambda c: sc.SHORT[c], key='d_class')
        mname = c2.radio('Model', ['HistGB', 'LogReg'], horizontal=True, key='d_model',
                         format_func=lambda m: 'Gradient boosting' if m == 'HistGB' else 'Logistic regression')
        e2 = explain_cached(df.iloc[[pos]][sc.FEATURES], held_out, mname)
        col = f'phi_{sc.SHORT[cls_choice]}'
        shown = e2['table'][['feature', 'meaning', 'value', 'unit', col]].rename(columns={col: 'contribution'})
        st.dataframe(shown, hide_index=True)
        st.caption(f'Average {sc.SHORT[cls_choice]} score over the reference sample {e2["base"][sc.SHORT[cls_choice]]:.2f}; '
                   f'this record {e2["fx"][sc.SHORT[cls_choice]]:.2f}. Contributions add up exactly '
                   f'(check {e2["additivity_error"]:.0e}).')
        if st.button('Global importance on this scenario', key='btn_global'):
            idx = np.linspace(0, n - 1, min(60, n)).astype(int)
            imp = global_cached(df.iloc[idx][sc.FEATURES].reset_index(drop=True), held_out, cls_choice)
            f2 = go.Figure(go.Bar(x=imp.values[::-1], y=list(imp.index[::-1]), orientation='h', marker_color=TEAL))
            f2.update_layout(template='plotly_dark', height=280, margin=dict(l=0, r=0, t=6, b=0),
                             xaxis_title=f'Mean |contribution| to the {sc.SHORT[cls_choice]} score (gradient boosting)',
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(f2)

    with t_models:
        d, p = df.iloc[:pos + 1], preds.iloc[:pos + 1]
        m1, m2 = st.columns(2)
        for cc, name, title in ((m1, 'HistGB', 'Gradient boosting'), (m2, 'LogReg', 'Logistic regression')):
            fig = go.Figure()
            for cls in (sc.CLASS_STABLE, sc.CLASS_TRANS, sc.CLASS_SEVERE):
                fig.add_trace(go.Scatter(x=d['TIME_h'], y=p[f'{name}_p_{sc.SHORT[cls]}'], mode='lines', stackgroup='one',
                                         name=sc.SHORT[cls], line=dict(width=0.5, color=CLASS_COL[cls]),
                                         fillcolor=CLASS_RGBA[cls]))
            fig.update_layout(template='plotly_dark', height=240, title=title, margin=dict(l=0, r=0, t=36, b=0),
                              yaxis=dict(range=[0, 1]), xaxis_title='Time (h)',
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            cc.plotly_chart(fig)
        st.caption(f'The models agree on {float(p["agree"].mean()):.0%} of the {len(p)} records shown.')

    with t_scen:
        share = pd.DataFrame({
            'Gradient boosting': preds['HistGB_class'].value_counts(normalize=True),
            'Logistic regression': preds['LogReg_class'].value_counts(normalize=True),
            'Proxy label': df['proxy_label'].value_counts(normalize=True)}).reindex(sc.CLASSES).fillna(0)
        fig = go.Figure()
        for cls in (sc.CLASS_STABLE, sc.CLASS_TRANS, sc.CLASS_SEVERE):
            vals = [float(share.loc[cls, c]) for c in share.columns]
            fig.add_trace(go.Bar(y=list(share.columns), x=vals, orientation='h', name=sc.SHORT[cls],
                                 marker_color=CLASS_COL[cls], text=[f'{v:.0%}' if v >= 0.06 else '' for v in vals],
                                 textposition='inside'))
        fig.update_layout(template='plotly_dark', barmode='stack', height=190, margin=dict(l=0, r=0, t=24, b=0),
                          xaxis=dict(range=[0, 1], visible=False), yaxis=dict(autorange='reversed'),
                          legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='left', x=0),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)
        st.caption('Whole scenario, all records. '
                   + ('In-sample: do not read agreement as performance.' if source == 'Demo scenarios' and held_out is None else ''))
        mt = st.radio('Leave-one-scenario-out check', ['All records', 'After the first 4.1 h'], horizontal=True, key='d_mt')
        if st.button('Run (about 10 s)', key='btn_loso'):
            tab = loso_cached(0.0 if mt == 'All records' else sc.STARTUP_HOURS)
            show = tab.copy()
            for c in ('dominant_proxy_class', 'HistGB', 'LogReg'):
                show[c] = show[c].map(sc.SHORT)
            st.dataframe(show, hide_index=True)
            st.markdown('  \n'.join(f'**{nm}**: dominant class right for {(tab[nm] == tab["dominant_proxy_class"]).sum()} '
                                    f'of {len(tab)} scenarios' for nm in ('HistGB', 'LogReg')))

    with t_log:
        d = pd.DataFrame({'record': np.arange(pos + 1), 'time_h': df['TIME_h'].iloc[:pos + 1].round(2).values,
                          'gradient_boosting': preds['HistGB_class'].iloc[:pos + 1].values,
                          'logistic_regression': preds['LogReg_class'].iloc[:pos + 1].values,
                          'data_quality': qual['status'].iloc[:pos + 1].values})
        d['priority'] = [sc.priority_for(a, b, c)[0] for a, b, c in
                         zip(d['gradient_boosting'], d['logistic_regression'], d['data_quality'])]
        key = d['gradient_boosting'] + '|' + d['logistic_regression'] + '|' + d['priority']
        ev = d[key != key.shift()].copy()
        ev['agreement'] = np.where(ev['gradient_boosting'] == ev['logistic_regression'], 'agree', 'disagree')
        ev['gradient_boosting'] = ev['gradient_boosting'].map(sc.SHORT)
        ev['logistic_regression'] = ev['logistic_regression'].map(sc.SHORT)
        st.dataframe(ev, hide_index=True)
        st.download_button('Download (CSV)', data=ev.to_csv(index=False).encode('utf-8'),
                           file_name='slugguard_event_log.csv', mime='text/csv')

    with t_about:
        st.markdown(
            '- **Classes** are a proxy label from swings in liquid flow and reversals of liquid velocity at the riser '
            'outlet. They are not OLGA flow regimes.\n'
            '- **Identification only.** Each record is classified on its own. Nothing forecasts a future event.\n'
            '- **Nine scenarios of one design case.** On unseen scenarios logistic regression named the dominant class '
            'in 7 of 9 cases and gradient boosting in 5 of 9.\n'
            '- **Scores are not calibrated probabilities.** Explanations are exact Shapley values over all 2⁷ input '
            'coalitions: they describe the model, not physical causes.\n'
            f'- **Models:** {model_source}. Start-up hours of a run may reflect start-up, not slugging.\n'
            '- **Read-only research prototype.** Not for operational decisions.')

# ---------------------------------------------------------------- replay loop
if st.session_state['playing']:
    if pos < n - 1:
        time.sleep(1.0 / speed)
        st.session_state['_advance'] = True
        st.rerun()
    else:
        st.session_state['playing'] = False
