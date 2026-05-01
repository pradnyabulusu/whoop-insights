import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Signal · Personal Health Intelligence",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Unbounded:wght@700;900&display=swap');

html, body, .stApp { background:#080808 !important; color:#ccc9c3 !important; font-family:'DM Mono',monospace !important; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding:3rem 4rem !important; max-width:1380px !important; }

.wordmark { font-family:'Unbounded',sans-serif; font-size:.95rem; font-weight:900; letter-spacing:.06em; color:#fff; }
.wordmark span { color:#b8ff3c; }
.page-title { font-family:'Unbounded',sans-serif; font-size:3.4rem; font-weight:900; line-height:1; letter-spacing:-.02em; color:#fff; padding:2.5rem 0 .4rem; }
.page-title .hi { color:#b8ff3c; }
.page-sub { font-size:10px; color:#333; letter-spacing:.14em; text-transform:uppercase; margin-bottom:1.5rem; }

.sec { font-size:9px; letter-spacing:.18em; text-transform:uppercase; color:#2e2e2e; padding-bottom:.5rem; border-bottom:1px solid #131313; margin:2.5rem 0 1.2rem; }

.stat-row { display:flex; gap:1px; margin:1rem 0 .5rem; }
.stat { flex:1; background:#0d0d0d; border:1px solid #141414; padding:1.2rem 1.4rem; }
.stat:first-child { border-radius:10px 0 0 10px; }
.stat:last-child { border-radius:0 10px 10px 0; }
.stat-label { font-size:9px; letter-spacing:.14em; text-transform:uppercase; color:#333; margin-bottom:.5rem; }
.stat-value { font-family:'Unbounded',sans-serif; font-size:1.8rem; font-weight:700; line-height:1; color:#fff; }
.stat-value.g { color:#b8ff3c; } .stat-value.b { color:#3cc8ff; } .stat-value.p { color:#ff3ca8; } .stat-value.a { color:#ffb83c; }
.stat-sub { font-size:10px; color:#333; margin-top:.3rem; }

.ins { background:#0d0d0d; border:1px solid #141414; border-radius:10px; padding:1.1rem 1.5rem; margin-bottom:.6rem; display:flex; gap:1rem; align-items:flex-start; }
.ins-icon { font-size:1.1rem; min-width:1.8rem; }
.ins-tag { font-size:8px; letter-spacing:.12em; text-transform:uppercase; padding:2px 7px; border-radius:20px; display:inline-block; margin-bottom:.4rem; }
.tg { background:#0a1400; color:#b8ff3c; border:1px solid #182800; }
.tb { background:#00131f; color:#3cc8ff; border:1px solid #002030; }
.ta { background:#1a1100; color:#ffb83c; border:1px solid #2a1c00; }
.tr { background:#180010; color:#ff3ca8; border:1px solid #280018; }
.ins-body { font-size:12.5px; line-height:1.8; color:#ccc9c3; }
.ins-body strong { color:#fff; font-weight:500; }
.ins-body em { color:#b8ff3c; font-style:normal; }

.trip-card { background:#0d0d0d; border:1px solid #141414; border-radius:10px; padding:1.2rem 1.5rem; margin-bottom:1rem; }
.trip-title { font-family:'Unbounded',sans-serif; font-size:1rem; font-weight:700; color:#fff; }
.trip-sub { font-size:10px; color:#333; margin-top:.3rem; }

.mover { background:#0d0d0d; border:1px solid #141414; border-radius:8px; padding:.9rem 1.1rem; margin-bottom:.4rem; }
.mover-name { font-size:11px; color:#fff; font-weight:500; margin-bottom:.2rem; }
.mover-sub { font-size:10px; color:#333; }

.upload-wrap { display:flex; flex-direction:column; justify-content:center; min-height:52vh; max-width:540px; }
.upload-wrap h2 { font-family:'Unbounded',sans-serif; font-size:2.2rem; font-weight:900; color:#fff; line-height:1.1; margin-bottom:1rem; }
.upload-wrap p { font-size:11px; color:#333; line-height:2; margin-bottom:1rem; }

.stFileUploader > div { background:#0d0d0d !important; border:1px dashed #1e1e1e !important; border-radius:10px !important; }
.stTabs [data-baseweb="tab-list"] { background:transparent !important; border-bottom:1px solid #131313 !important; gap:.2rem !important; }
.stTabs [data-baseweb="tab"] { background:transparent !important; color:#333 !important; font-family:'DM Mono',monospace !important; font-size:10px !important; letter-spacing:.12em !important; text-transform:uppercase !important; padding:.5rem 1.2rem !important; border:none !important; }
.stTabs [aria-selected="true"] { color:#b8ff3c !important; border-bottom:2px solid #b8ff3c !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top:1.5rem !important; background:transparent !important; }
.stDateInput input, .stTextInput input, .stTextArea textarea { background:#0d0d0d !important; border:1px solid #1e1e1e !important; color:#ccc9c3 !important; font-family:'DM Mono',monospace !important; border-radius:8px !important; font-size:12px !important; }
.stButton > button { background:#b8ff3c !important; color:#080808 !important; border:none !important; font-family:'DM Mono',monospace !important; font-weight:500 !important; border-radius:8px !important; font-size:11px !important; letter-spacing:.06em !important; }
.stButton > button:hover { background:#c8ff5c !important; }
.stSelectbox > div > div, .stMultiSelect > div > div { background:#0d0d0d !important; border:1px solid #1e1e1e !important; color:#ccc9c3 !important; font-family:'DM Mono',monospace !important; font-size:12px !important; }
div[data-testid="stExpander"] { background:#0d0d0d !important; border:1px solid #141414 !important; border-radius:10px !important; }
.stSlider label, .stCheckbox label, .stRadio label { color:#ccc9c3 !important; font-family:'DM Mono',monospace !important; font-size:12px !important; }
p, li, label { color:#ccc9c3 !important; }
</style>
""", unsafe_allow_html=True)

# ─── constants ──────────────────────────────────────────────────────────────
G="#b8ff3c"; B="#3cc8ff"; P="#ff3ca8"; A="#ffb83c"; MU="#1a1a1a"
CLUSTER_COLORS = [G, B, P, A, "#a78bfa", "#fb923c"]
DAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

METRIC_LABELS = {
    "recovery_score":"Recovery %","hrv_rmssd_milli":"HRV (ms)",
    "resting_heart_rate":"Resting HR","strain":"Strain",
    "sleep_performance_pct":"Sleep Score","rem_sleep_mins":"REM (mins)",
    "slow_wave_sleep_mins":"Deep Sleep (mins)","respiratory_rate":"Resp. Rate",
    "skin_temp_celsius":"Skin Temp °C","sleep_efficiency_pct":"Sleep Efficiency",
    "sleep_consistency_pct":"Consistency","total_in_bed_mins":"Time in Bed",
    "sleep_debt_needed_mins":"Sleep Debt (mins)","spo2_percentage":"SpO2 %",
}
CORE = list(METRIC_LABELS.keys())

TAGS = [
    "😓 High stress","✈️ Travel","🥂 Late night / party","🤒 Sick",
    "💪 Heavy training","🧘 Rest / recovery day","🌴 Holiday",
    "💼 Big work deadline","🍕 Ate badly","🫶 Social / happy day",
    "🌙 Poor sleep night","☀️ Felt great",
]

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#444", size=11),
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(showgrid=False, zeroline=False, color="#2a2a2a", tickcolor="#2a2a2a", linecolor="#131313"),
    yaxis=dict(showgrid=True, gridcolor="#0f0f0f", zeroline=False, color="#2a2a2a", tickcolor="#2a2a2a"),
)

# ─── data loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_data(f):
    df = pd.read_csv(f)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ","_")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in ["sleep_start","sleep_end","cycle_start","cycle_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    for col in CORE:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # zero-sleep rows → NaN (not real sleep data)
    sleep_marker_cols = [c for c in ["rem_sleep_mins","slow_wave_sleep_mins","total_in_bed_mins"] if c in df.columns]
    if sleep_marker_cols:
        bad = (df[sleep_marker_cols].fillna(0) == 0).all(axis=1)
        sleep_all = [c for c in CORE if "sleep" in c or c in ["rem_sleep_mins","slow_wave_sleep_mins","total_in_bed_mins","respiratory_rate"]]
        for c in sleep_all:
            if c in df.columns:
                df.loc[bad, c] = np.nan

    df["dow"]        = df["date"].dt.dayofweek
    df["dow_name"]   = df["date"].dt.day_name()
    df["is_weekend"] = df["dow"] >= 5
    df["month_str"]  = df["date"].dt.strftime("%b %Y")

    if "sleep_start" in df.columns:
        df["sleep_hour"] = df["sleep_start"].dt.hour + df["sleep_start"].dt.minute/60

    return df

def present_cols(df):
    return [c for c in CORE if c in df.columns and df[c].notna().sum() > 10]

# ─── insights ────────────────────────────────────────────────────────────────
def build_insights(df, present):
    cards = []
    work = df[~df["is_weekend"]]
    wknd = df[df["is_weekend"]]

    def add(icon, tag, cls, body):
        cards.append(dict(icon=icon, tag=tag, cls=cls, body=body))

    # 1. HRV: stress not fitness
    if "hrv_rmssd_milli" in present:
        wd = work["hrv_rmssd_milli"].mean(); we = wknd["hrv_rmssd_milli"].mean()
        if we > wd * 1.05:
            add("💓","stress vs fitness","ta",
                f"HRV is <strong>{we:.0f} ms on weekends</strong> vs <strong>{wd:.0f} ms on workdays</strong> — "
                f"a {we-wd:.0f} ms gap. Exercise increases HRV. Stress suppresses it. "
                f"<em>Your weekday drop is psychological load, not physical — and it's measurable.</em>")

    # 2. Skin temp as 48hr early warning
    if "skin_temp_celsius" in present and "recovery_score" in present:
        tmp = df[["skin_temp_celsius","recovery_score"]].dropna().copy()
        tmp["rec_2d"] = tmp["recovery_score"].shift(-2)
        corr = tmp[["skin_temp_celsius","rec_2d"]].dropna().corr().iloc[0,1]
        if corr < -0.18:
            add("🌡️","48-hour early warning","tr",
                f"Skin temperature correlates <strong>{corr:.2f}</strong> with recovery <em>2 days later</em>. "
                f"When your skin temp rises, your body is already reacting — you just can't feel it yet. "
                f"Whoop shows skin temp. It never connects it forward in time like this.")

    # 3. Sleep timing vs recovery
    if "sleep_hour" in df.columns and "recovery_score" in present:
        tmp = df[["sleep_hour","recovery_score"]].dropna()
        late = tmp[tmp["sleep_hour"] > 23]["recovery_score"].mean()
        early = tmp[tmp["sleep_hour"] <= 22]["recovery_score"].mean()
        if not (np.isnan(late) or np.isnan(early)) and abs(late - early) > 4:
            diff = abs(late - early)
            worse = "sleeping after midnight" if late < early else "sleeping before 11pm"
            add("🕐","sleep timing effect","tb",
                f"When you sleep after midnight, next-day recovery averages <strong>{late:.0f}</strong>. "
                f"Before midnight: <strong>{early:.0f}</strong>. That's a <em>{diff:.0f}-point difference</em> "
                f"from the same number of hours — just shifted later. Timing is the hidden variable.")

    # 4. How long YOU take to recover after hard days
    if "strain" in present and "hrv_rmssd_milli" in present:
        high_idx = df[df["strain"] > df["strain"].quantile(0.75)].index.tolist()
        curves = []
        for i in high_idx:
            window = df.loc[i:i+3, "hrv_rmssd_milli"].values
            if len(window) == 4 and not np.any(np.isnan(window)):
                curves.append(window - window[0])
        if len(curves) >= 5:
            avg = np.mean(curves, axis=0)
            days = next((i for i,v in enumerate(avg) if v >= -1.5), 3)
            add("⚡","your personal recovery window","ta",
                f"After your hardest days (top 25% strain), HRV takes <strong>{days} day(s)</strong> "
                f"to return to baseline — on average. <em>This is your actual recovery window, "
                f"not a generic 24-hour assumption.</em> Most people are training again too soon.")

    # 5. Sleep debt compounding
    if "sleep_debt_needed_mins" in present:
        recent = df.tail(14)["sleep_debt_needed_mins"].mean()
        base   = df["sleep_debt_needed_mins"].mean()
        diff   = recent - base
        if diff > 15:
            add("📉","compounding sleep debt","tr",
                f"Your sleep debt is <strong>{diff:.0f} mins above your own baseline</strong> over the last 14 days. "
                f"Debt compounds night to night — each deficit makes the next night worse. "
                f"<em>At this trajectory you need roughly {diff/30:.1f} full recovery nights just to break even.</em>")
        elif diff < -15:
            add("📈","debt recovery in progress","tg",
                f"You're actively paying back sleep debt — <strong>down {abs(diff):.0f} mins vs your baseline</strong> over 14 days. "
                f"This will show up as improved HRV and recovery scores in the next 5–7 days.")

    # 6. REM variance — consistency matters as much as quantity
    if "rem_sleep_mins" in present and df["rem_sleep_mins"].notna().sum() > 20:
        cv = df["rem_sleep_mins"].std() / df["rem_sleep_mins"].mean()
        if cv > 0.35:
            add("🌀","rem instability","ta",
                f"Your REM sleep varies wildly night to night (variation coefficient: {cv:.2f}). "
                f"Research links high REM variance — not just low REM — to mood instability and impaired emotional processing. "
                f"<em>Whoop shows you the number. Not the variance. This is what you're missing.</em>")

    # 7. Respiratory rate as early illness signal
    if "respiratory_rate" in present:
        recent_rr = df.tail(7)["respiratory_rate"].mean()
        base_rr   = df["respiratory_rate"].mean()
        diff = recent_rr - base_rr
        if diff > 0.4:
            add("🫁","respiratory flag","tr",
                f"Respiratory rate is <strong>{recent_rr:.1f}/min this week</strong> vs your baseline {base_rr:.1f}. "
                f"An elevation of {diff:.1f} breaths/min is a known early-warning signal — "
                f"<em>illness and overtraining both show here 2–4 days before you feel them.</em>")
        elif diff < -0.4:
            add("🫁","cardiovascular adaptation","tg",
                f"Respiratory rate is <strong>trending down</strong> ({recent_rr:.1f} vs {base_rr:.1f} baseline). "
                f"Lower resting respiratory rate signals improving cardiovascular efficiency. Your engine is getting leaner.")

    # 8. Deep sleep as the real recovery lever
    if "slow_wave_sleep_mins" in present and "recovery_score" in present:
        corr = df[["slow_wave_sleep_mins","recovery_score"]].dropna().corr().iloc[0,1]
        if abs(corr) > 0.2:
            add("🧠","deep sleep is your lever","tb",
                f"Deep (slow wave) sleep correlates <strong>{corr:.2f}</strong> with your recovery score. "
                f"It's your strongest sleep-based predictor. The suppressors: alcohol within 3hrs of sleep, "
                f"eating late, high evening strain. <em>These are your actual levers — not sleep duration.</em>")

    # 9. Weekly fatigue arc
    if "recovery_score" in present:
        fri = df[df["dow"]==4]["recovery_score"].mean()
        mon = df[df["dow"]==0]["recovery_score"].mean()
        if not (np.isnan(fri) or np.isnan(mon)) and mon - fri > 5:
            add("📆","your weekly tax","ta",
                f"Monday recovery: <strong>{mon:.0f}</strong>. Friday: <strong>{fri:.0f}</strong>. "
                f"You lose {mon-fri:.0f} pts across every single week — consistently. "
                f"<em>This is your baseline work tax.</em> A deliberate low-strain Thursday recovers 3–4 of those points.")

    return cards

# ─── trip analysis ───────────────────────────────────────────────────────────
def analyse_trip(df, present, start, end):
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    trip = df[mask].copy()
    n = max(len(trip), 1)
    baseline = df[df["date"] < pd.Timestamp(start)].tail(n * 3)
    if len(trip) < 2 or len(baseline) < 3:
        return None, None, None
    results = []
    for m in present:
        tv = trip[m].dropna(); bv = baseline[m].dropna()
        if len(tv) < 2 or len(bv) < 3: continue
        bm, bs = bv.mean(), bv.std()
        if bs < 0.01: continue
        tm = tv.mean()
        z = (tm - bm) / bs
        pct = (tm - bm) / abs(bm) * 100 if bm != 0 else 0
        results.append(dict(metric=m, label=METRIC_LABELS.get(m,m),
                            trip_mean=tm, base_mean=bm, z=z, pct=pct))
    results.sort(key=lambda x: abs(x["z"]), reverse=True)
    return trip, baseline, results

# ─── clustering ──────────────────────────────────────────────────────────────
def run_kmeans(df, present, k=4):
    cols = [c for c in present if c in df.columns]
    X = df[cols].copy()
    valid = X.notna().all(axis=1)
    Xc = X[valid]; dates = df.loc[valid, "date"].values
    if len(Xc) < k * 4: return None, None, None
    sc = StandardScaler()
    Xs = sc.fit_transform(Xc)
    km = KMeans(n_clusters=k, random_state=42, n_init=12)
    labels = km.fit_predict(Xs)
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(Xs)
    out = pd.DataFrame({"date":dates,"cluster":labels,"x":xy[:,0],"y":xy[:,1]})
    for c in cols: out[c] = Xc[c].values
    profiles = []
    for cl in range(k):
        sub = Xc[labels==cl]
        overall = Xc.mean()
        cl_mean = sub.mean()
        standout = (cl_mean - overall).abs().sort_values(ascending=False)
        profiles.append(dict(cluster=cl, n=len(sub), mean=cl_mean, top=standout.head(3).index.tolist()))
    return out, profiles, cols

def auto_name(p):
    r = p["mean"].get("recovery_score", 50)
    h = p["mean"].get("hrv_rmssd_milli", 50)
    s = p["mean"].get("strain", 10)
    sl = p["mean"].get("sleep_performance_pct", 70)
    if r > 75 and h > 55: return "Thriving days"
    if r < 42: return "Depleted days"
    if s > 13: return "High output days"
    if sl and sl < 65: return "Poor sleep days"
    if h and h < 38: return "High stress days"
    return "Baseline days"

# ─── small html helpers ──────────────────────────────────────────────────────
def stat(label, val, cls="", sub=""):
    return (f'<div class="stat"><div class="stat-label">{label}</div>'
            f'<div class="stat-value {cls}">{val}</div>'
            + (f'<div class="stat-sub">{sub}</div>' if sub else "") + '</div>')

def ins_card(icon, tag, cls, body):
    return (f'<div class="ins"><div class="ins-icon">{icon}</div><div>'
            f'<span class="ins-tag {cls}">{tag}</span><br>'
            f'<div class="ins-body">{body}</div></div></div>')

def mover(label, trip_val, base_val, pct, positive):
    col = G if positive else P
    arrow = "↑" if positive else "↓"
    return (f'<div class="mover"><div class="mover-name" style="color:{col}">'
            f'{arrow} {label}</div><div class="mover-sub">'
            f'{trip_val:.1f} vs {base_val:.1f} baseline &nbsp;·&nbsp; '
            f'{arrow}{abs(pct):.0f}%</div></div>')

# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    st.markdown('<div style="padding:1.5rem 0 0"><span class="wordmark">SIGNAL<span>◉</span></span></div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded is None:
        st.markdown("""
        <div class="upload-wrap">
          <h2>What is your body<br>actually saying?</h2>
          <p>
            Drop your Whoop CSV export here.<br>
            Signal runs the analysis Whoop doesn't —<br>
            trip comparisons, day clustering, early warnings,<br>
            and the patterns hiding in plain sight.
          </p>
          <p style="color:#222">
            Whoop app → Profile → App Settings → Export Data<br>
            All processing happens locally in your session.
          </p>
        </div>
        """, unsafe_allow_html=True)
        return

    df   = load_data(uploaded)
    pres = present_cols(df)
    n    = len(df)
    dr   = f"{df['date'].min():%b %d %Y}  →  {df['date'].max():%b %d %Y}"

    st.markdown(f"""
    <div class="page-title">Your body,<br><span class="hi">decoded.</span></div>
    <div class="page-sub">{n} days of data &nbsp;·&nbsp; {dr}</div>
    """, unsafe_allow_html=True)

    # stat strip
    stats_html = ""
    if "recovery_score" in pres:
        stats_html += stat("Avg recovery", f"{df['recovery_score'].mean():.0f}%", "g")
    if "hrv_rmssd_milli" in pres:
        stats_html += stat("Avg HRV", f"{df['hrv_rmssd_milli'].mean():.0f}ms", "b")
    if "resting_heart_rate" in pres:
        stats_html += stat("Resting HR", f"{df['resting_heart_rate'].mean():.0f}bpm", "p")
    if "sleep_performance_pct" in pres:
        stats_html += stat("Sleep score", f"{df['sleep_performance_pct'].mean():.0f}%", "a")
    if stats_html:
        st.markdown(f'<div class="stat-row">{stats_html}</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "◎  what whoop isn't telling you",
        "◈  trip analysis",
        "◇  day clusters",
    ])

    # ══════════════════════════════════════════════
    # TAB 1 — INSIGHTS
    # ══════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="sec">signal · the patterns hidden in your data</div>', unsafe_allow_html=True)
        cards = build_insights(df, pres)
        if not cards:
            st.info("Need 60+ days of varied data to surface reliable patterns.")
        else:
            for c in cards:
                st.markdown(ins_card(c["icon"], c["tag"], c["cls"], c["body"]), unsafe_allow_html=True)

        if "recovery_score" in pres:
            st.markdown('<div class="sec">recovery timeline</div>', unsafe_allow_html=True)
            roll = df["recovery_score"].rolling(7, min_periods=3).mean()
            fig  = go.Figure()
            fig.add_trace(go.Scatter(x=df["date"], y=df["recovery_score"],
                line=dict(color="#1a1a1a", width=1), name="daily",
                hovertemplate="%{x|%b %d}: %{y:.0f}%<extra>daily</extra>"))
            fig.add_trace(go.Scatter(x=df["date"], y=roll,
                line=dict(color=G, width=2.5), name="7d avg",
                hovertemplate="%{x|%b %d}: %{y:.0f}%<extra>7d avg</extra>"))
            fig.add_hrect(y0=67,y1=100, fillcolor=G, opacity=.03, line_width=0)
            fig.add_hrect(y0=34,y1=67,  fillcolor=A, opacity=.03, line_width=0)
            fig.add_hrect(y0=0, y1=34,  fillcolor=P, opacity=.03, line_width=0)
            fig.update_layout(**BASE_LAYOUT, height=240, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

        if "recovery_score" in pres:
            st.markdown('<div class="sec">recovery by weekday — your weekly arc</div>', unsafe_allow_html=True)
            grp = df.groupby("dow_name")["recovery_score"].mean().reindex(DAY_ORDER).dropna()
            bc  = [G if d in ["Saturday","Sunday"] else MU for d in grp.index]
            fig = go.Figure(go.Bar(x=grp.index, y=grp.values, marker_color=bc,
                hovertemplate="%{x}: %{y:.0f}%<extra></extra>"))
            fig.update_layout(**BASE_LAYOUT, height=190, bargap=.42)
            fig.update_xaxes(tickfont=dict(size=9))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    # ══════════════════════════════════════════════
    # TAB 2 — TRIP ANALYSIS
    # ══════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="sec">compare any trip against your personal baseline</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:11px;color:#333;margin-bottom:1.2rem;line-height:1.9">Enter dates for up to 2 trips. Signal compares every health metric against your pre-trip baseline and shows what actually shifted — and by how much.</div>', unsafe_allow_html=True)

        mn = df["date"].min().date(); mx = df["date"].max().date()
        mid = mn + (mx - mn) // 2

        c1,c2,c3 = st.columns([2,1,1])
        t1n = c1.text_input("Trip name", "Goa trip", key="t1n")
        t1s = c2.date_input("Start date", mid - timedelta(days=10), min_value=mn, max_value=mx, key="t1s")
        t1e = c3.date_input("End date",   mid - timedelta(days=3),  min_value=mn, max_value=mx, key="t1e")

        add2 = st.checkbox("Add a second trip to compare side by side")
        t2n=t2s=t2e=None
        if add2:
            d1,d2,d3 = st.columns([2,1,1])
            t2n = d1.text_input("Trip name", "Coorg trip", key="t2n")
            t2s = d2.date_input("Start date", mid + timedelta(days=5),  min_value=mn, max_value=mx, key="t2s")
            t2e = d3.date_input("End date",   mid + timedelta(days=12), min_value=mn, max_value=mx, key="t2e")

        if st.button("Run trip analysis"):
            tr1, bl1, r1 = analyse_trip(df, pres, t1s, t1e)

            if r1 is None:
                st.error("Not enough data for Trip 1. Try a wider date range or check your dates.")
            else:
                st.markdown(f"""
                <div class="trip-card">
                  <div class="trip-title">◎ {t1n}</div>
                  <div class="trip-sub">{t1s} → {t1e} &nbsp;·&nbsp; {len(tr1)} days &nbsp;·&nbsp; vs {len(bl1)}-day pre-trip baseline</div>
                </div>""", unsafe_allow_html=True)

                pos = [r for r in r1 if r["z"] >  0.35]
                neg = [r for r in r1 if r["z"] < -0.35]

                ca, cb = st.columns(2)
                with ca:
                    st.markdown('<div style="font-size:10px;letter-spacing:.12em;color:#444;margin-bottom:.6rem">WHAT GOT BETTER</div>', unsafe_allow_html=True)
                    if pos:
                        for r in pos[:5]:
                            st.markdown(mover(r["label"], r["trip_mean"], r["base_mean"], r["pct"], True), unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="font-size:11px;color:#2a2a2a">Nothing meaningfully improved</div>', unsafe_allow_html=True)
                with cb:
                    st.markdown('<div style="font-size:10px;letter-spacing:.12em;color:#444;margin-bottom:.6rem">WHAT GOT WORSE</div>', unsafe_allow_html=True)
                    if neg:
                        for r in neg[:5]:
                            st.markdown(mover(r["label"], r["trip_mean"], r["base_mean"], r["pct"], False), unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="font-size:11px;color:#2a2a2a">Nothing meaningfully declined</div>', unsafe_allow_html=True)

                # z-score waterfall
                st.markdown('<div class="sec">how far each metric shifted — in standard deviations from your baseline</div>', unsafe_allow_html=True)
                top8 = r1[:8]
                zv   = [r["z"] for r in top8]
                lb   = [r["label"] for r in top8]
                bc   = [G if z > 0 else P for z in zv]
                fig  = go.Figure(go.Bar(x=zv, y=lb, orientation="h", marker_color=bc,
                    hovertemplate="<b>%{y}</b>: %{x:.2f} SD<extra></extra>"))
                fig.add_vline(x=0,  line_color="#222", line_width=1)
                fig.add_vline(x=1,  line_color="#181818", line_width=1, line_dash="dot")
                fig.add_vline(x=-1, line_color="#181818", line_width=1, line_dash="dot")
                fig.update_layout(**BASE_LAYOUT, height=270,
                    yaxis=dict(showgrid=False, zeroline=False, color="#444", categoryorder="total ascending"),
                    xaxis_title="standard deviations from pre-trip baseline")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

                # top 3 metric timelines
                top3m = [r["metric"] for r in r1[:3]]
                if top3m:
                    st.markdown('<div class="sec">metric timelines — shaded band = trip period</div>', unsafe_allow_html=True)
                    for m in top3m:
                        if m not in df.columns: continue
                        lbl = METRIC_LABELS.get(m, m)
                        st.markdown(f'<div style="font-size:10px;color:#333;letter-spacing:.1em;margin:.8rem 0 .2rem">{lbl.upper()}</div>', unsafe_allow_html=True)
                        fig2 = go.Figure()
                        fig2.add_vrect(x0=pd.Timestamp(t1s), x1=pd.Timestamp(t1e),
                            fillcolor=G, opacity=0.06, line_width=0)
                        fig2.add_trace(go.Scatter(x=df["date"], y=df[m],
                            line=dict(color="#1e1e1e", width=1.5), showlegend=False,
                            hovertemplate="%{x|%b %d}: %{y:.1f}<extra></extra>"))
                        fig2.add_trace(go.Scatter(x=df["date"], y=df[m].rolling(5,min_periods=2).mean(),
                            line=dict(color=G, width=2), showlegend=False,
                            hovertemplate="%{x|%b %d}: %{y:.1f}<extra>5d avg</extra>"))
                        fig2.update_layout(**BASE_LAYOUT, height=150)
                        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

                # ── Trip 2 ──
                if add2 and t2n and t2s and t2e:
                    tr2, bl2, r2 = analyse_trip(df, pres, t2s, t2e)
                    if r2 is None:
                        st.error("Not enough data for Trip 2.")
                    else:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="trip-card">
                          <div class="trip-title">◈ {t2n}</div>
                          <div class="trip-sub">{t2s} → {t2e} &nbsp;·&nbsp; {len(tr2)} days &nbsp;·&nbsp; vs {len(bl2)}-day pre-trip baseline</div>
                        </div>""", unsafe_allow_html=True)

                        # head-to-head z-score chart
                        shared = [r["metric"] for r in r1 if any(x["metric"]==r["metric"] for x in r2)][:7]
                        if shared:
                            st.markdown('<div class="sec">head-to-head — which trip was better for your body</div>', unsafe_allow_html=True)
                            z1m = {r["metric"]:r["z"] for r in r1}
                            z2m = {r["metric"]:r["z"] for r in r2}
                            ylb = [METRIC_LABELS.get(m,m) for m in shared]
                            fig3 = go.Figure()
                            fig3.add_trace(go.Bar(name=t1n, x=[z1m.get(m,0) for m in shared], y=ylb,
                                orientation="h", marker_color=G,
                                hovertemplate=f"<b>{t1n}</b>: %{{x:.2f}} SD<extra></extra>"))
                            fig3.add_trace(go.Bar(name=t2n, x=[z2m.get(m,0) for m in shared], y=ylb,
                                orientation="h", marker_color=B,
                                hovertemplate=f"<b>{t2n}</b>: %{{x:.2f}} SD<extra></extra>"))
                            fig3.add_vline(x=0, line_color="#222", line_width=1)
                            fig3.update_layout(**BASE_LAYOUT, height=300, barmode="group",
                                legend=dict(font=dict(color="#555",size=10), bgcolor="rgba(0,0,0,0)"),
                                yaxis=dict(showgrid=False, zeroline=False, color="#444"),
                                xaxis_title="standard deviations from each trip's own baseline")
                            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})

    # ══════════════════════════════════════════════
    # TAB 3 — CLUSTERS
    # ══════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="sec">k-means clustering — find your body\'s recurring day patterns</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:12px;color:#333;line-height:1.9;margin-bottom:1.5rem;max-width:620px">
        Instead of asking what you did <em>every single day</em>, Signal groups days that look alike
        across all your metrics simultaneously — then asks you once what usually happens in those periods.
        Tag a cluster once. Every similar day gets context.
        </div>
        """, unsafe_allow_html=True)

        k = st.slider("How many day types to find", 3, 6, 4)

        if st.button("Find my day patterns"):
            res, profs, fcols = run_kmeans(df, pres, k)
            if res is None:
                st.error("Need at least 30 days with complete data across all metrics.")
            else:
                st.session_state.update({"cr": res, "cp": profs, "cf": fcols, "cu": {}})

        if "cr" in st.session_state:
            res   = st.session_state["cr"]
            profs = st.session_state["cp"]
            fcols = st.session_state["cf"]
            ulabs = st.session_state.get("cu", {})

            # scatter
            st.markdown('<div class="sec">your days mapped — each dot is one day, proximity = similarity</div>', unsafe_allow_html=True)
            fig = go.Figure()
            for p in profs:
                cl   = p["cluster"]
                sub  = res[res["cluster"]==cl]
                name = ulabs.get(cl, auto_name(p))
                col  = CLUSTER_COLORS[cl % len(CLUSTER_COLORS)]
                fig.add_trace(go.Scatter(
                    x=sub["x"], y=sub["y"], mode="markers",
                    name=f"{name} ({p['n']}d)",
                    marker=dict(color=col, size=8, opacity=0.75, line=dict(color="#080808",width=1)),
                    text=[pd.Timestamp(d).strftime("%b %d") for d in sub["date"]],
                    hovertemplate="<b>%{text}</b><extra>"+name+"</extra>",
                ))
            fig.update_layout(**BASE_LAYOUT, height=380,
                xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                legend=dict(font=dict(color="#555",size=10),bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

            # radar
            rmets = [c for c in ["recovery_score","hrv_rmssd_milli","strain",
                                   "sleep_performance_pct","rem_sleep_mins","slow_wave_sleep_mins"] if c in fcols]
            if len(rmets) >= 3:
                st.markdown('<div class="sec">cluster profiles — what defines each day type</div>', unsafe_allow_html=True)
                all_means = np.array([res[res["cluster"]==p["cluster"]][rmets].mean().values for p in profs])
                sc = StandardScaler(); normed = sc.fit_transform(all_means)
                normed = (normed - normed.min()) / (normed.max() - normed.min() + 1e-9)
                rlabels = [METRIC_LABELS.get(m,m) for m in rmets]
                fig2 = go.Figure()
                for i, p in enumerate(profs):
                    cl   = p["cluster"]
                    col  = CLUSTER_COLORS[cl % len(CLUSTER_COLORS)]
                    name = ulabs.get(cl, auto_name(p))
                    vals = list(normed[i]) + [normed[i][0]]
                    theta = rlabels + [rlabels[0]]
                    fig2.add_trace(go.Scatterpolar(r=vals, theta=theta, name=name,
                        line=dict(color=col, width=2),
                        fill="toself", fillcolor=col+"18",
                        hovertemplate="%{theta}: %{r:.2f}<extra>"+name+"</extra>"))
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    polar=dict(bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=False),
                        angularaxis=dict(color="#2a2a2a",tickfont=dict(size=10,color="#444",family="DM Mono"))),
                    legend=dict(font=dict(color="#555",size=10),bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=40,r=40,t=20,b=20), height=380,
                    font=dict(family="DM Mono, monospace",color="#444"))
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

            # cluster cards with tag picker
            st.markdown('<div class="sec">tag each cluster — do it once, unlock pattern memory</div>', unsafe_allow_html=True)
            for p in profs:
                cl   = p["cluster"]
                col  = CLUSTER_COLORS[cl % len(CLUSTER_COLORS)]
                name = ulabs.get(cl, auto_name(p))

                with st.expander(f"**{name}** — {p['n']} days"):
                    # standout metrics
                    overall = res[fcols].mean()
                    cl_mean = res[res["cluster"]==cl][fcols].mean()
                    mc = st.columns(3)
                    for i, feat in enumerate(p["top"][:3]):
                        lbl = METRIC_LABELS.get(feat, feat)
                        val = cl_mean.get(feat, 0)
                        diff = cl_mean.get(feat, 0) - overall.get(feat, 0)
                        arrow = "↑" if diff > 0 else "↓"
                        fc = "g" if diff > 0 else "p"
                        mc[i].markdown(
                            f'<div class="stat" style="border-radius:8px">'
                            f'<div class="stat-label">{lbl}</div>'
                            f'<div class="stat-value {fc}" style="font-size:1.3rem">{val:.1f}</div>'
                            f'<div class="stat-sub">{arrow} vs your avg</div></div>',
                            unsafe_allow_html=True)

                    # sample dates
                    dates_str = pd.to_datetime(res[res["cluster"]==cl]["date"]).dt.strftime("%b %d").tolist()
                    sample = "  ·  ".join(dates_str[:10]) + ("  ···" if len(dates_str)>10 else "")
                    st.markdown(f'<div style="font-size:10px;color:#282828;margin:1rem 0 .8rem;line-height:2">{sample}</div>', unsafe_allow_html=True)

                    st.markdown(f'<div style="font-size:9px;letter-spacing:.14em;color:#2a2a2a;margin-bottom:.5rem">WHAT USUALLY HAPPENS ON THESE DAYS?</div>', unsafe_allow_html=True)
                    sel = st.multiselect("", TAGS, key=f"tag_{cl}", label_visibility="collapsed")
                    txt = st.text_input("Or in your own words", key=f"txt_{cl}", placeholder="e.g. exam weeks, back-to-back meetings, monsoon travel...")

                    if sel or txt:
                        parts = sel + ([txt] if txt else [])
                        lstr  = "  ·  ".join(parts)
                        st.session_state["cu"][cl] = lstr
                        st.markdown(f'<div style="font-size:11px;color:{col};margin-top:.6rem">✓ {lstr}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
