import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Body Intelligence",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── design system ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Syne:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #0a0a0a;
  --surface: #111111;
  --surface2: #1a1a1a;
  --border: #222222;
  --accent: #c8f060;
  --accent2: #60d4f0;
  --accent3: #f060a8;
  --text: #f0ede8;
  --muted: #888888;
  --warn: #f0b060;
}

html, body, .stApp { background: var(--bg) !important; color: var(--text) !important; }

.stApp { font-family: 'DM Mono', monospace !important; }

h1, h2, h3, .syne { font-family: 'Syne', sans-serif !important; }

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }

/* upload zone */
.stFileUploader > div {
  background: var(--surface) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 12px !important;
}
.stFileUploader label { color: var(--muted) !important; font-family: 'DM Mono', monospace !important; }

/* selectbox / multiselect */
.stSelectbox > div > div, .stMultiSelect > div > div {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: 'DM Mono', monospace !important;
}

/* metric cards */
.metric-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.4rem 1.6rem;
  position: relative;
  overflow: hidden;
  transition: border-color .2s;
}
.metric-card:hover { border-color: #333; }
.metric-card .label {
  font-size: 10px;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: .4rem;
}
.metric-card .value {
  font-family: 'Syne', sans-serif;
  font-size: 2.2rem;
  font-weight: 700;
  line-height: 1;
  color: var(--text);
}
.metric-card .delta {
  font-size: 11px;
  margin-top: .5rem;
  color: var(--muted);
}
.metric-card .accent-bar {
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 3px;
  border-radius: 3px 0 0 3px;
}

/* insight cards */
.insight-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.2rem 1.4rem;
  margin-bottom: .8rem;
  display: flex;
  gap: 1rem;
  align-items: flex-start;
}
.insight-icon {
  font-size: 1.3rem;
  min-width: 2rem;
  margin-top: .05rem;
}
.insight-text { font-size: 13px; line-height: 1.6; color: var(--text); }
.insight-text strong { color: var(--accent); font-weight: 500; }
.insight-tag {
  display: inline-block;
  font-size: 9px;
  letter-spacing: .1em;
  text-transform: uppercase;
  padding: 2px 8px;
  border-radius: 20px;
  margin-bottom: .4rem;
}
.tag-warn { background: #2a1f00; color: var(--warn); border: 1px solid #3a2900; }
.tag-good { background: #1a2a00; color: var(--accent); border: 1px solid #2a3a00; }
.tag-info { background: #002a2f; color: var(--accent2); border: 1px solid #003a45; }

/* section headers */
.section-header {
  font-family: 'Syne', sans-serif;
  font-size: 11px;
  letter-spacing: .15em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 2.5rem 0 1rem;
  padding-bottom: .5rem;
  border-bottom: 1px solid var(--border);
}

/* hero */
.hero {
  padding: 3rem 0 1.5rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2rem;
}
.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: 3.5rem;
  font-weight: 800;
  line-height: 1;
  letter-spacing: -.02em;
  color: var(--text);
  margin-bottom: .5rem;
}
.hero-sub {
  font-size: 12px;
  color: var(--muted);
  letter-spacing: .05em;
}
.hero-accent { color: var(--accent); }

/* upload screen */
.upload-hero {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 60vh;
  text-align: center;
  gap: 1rem;
}
.upload-title {
  font-family: 'Syne', sans-serif;
  font-size: 2.8rem;
  font-weight: 800;
  color: var(--text);
  line-height: 1.1;
}
.upload-sub { font-size: 12px; color: var(--muted); max-width: 420px; line-height: 1.7; }
.dot { color: var(--accent); }

/* tabs */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: .5rem !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: .08em !important;
  border: none !important;
  padding: .5rem 1rem !important;
}
.stTabs [aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; background: transparent !important; }

/* divider */
hr { border-color: var(--border) !important; }

/* slider */
.stSlider > div { color: var(--muted) !important; }
</style>
""", unsafe_allow_html=True)

# ── colour helpers ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#888888", size=11),
    xaxis=dict(showgrid=False, zeroline=False, color="#444", tickcolor="#444"),
    yaxis=dict(showgrid=True, gridcolor="#1e1e1e", zeroline=False, color="#444", tickcolor="#444"),
    margin=dict(l=0, r=0, t=20, b=0),
)
ACCENT  = "#c8f060"
ACCENT2 = "#60d4f0"
ACCENT3 = "#f060a8"
WARN    = "#f0b060"
MUTED   = "#444444"


# ── data loading ───────────────────────────────────────────────────────────────
def load_data(f) -> pd.DataFrame:
    df = pd.read_csv(f)
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["dow"]        = df["date"].dt.dayofweek          # 0=Mon … 6=Sun
    df["dow_name"]   = df["date"].dt.day_name()
    df["is_weekend"] = df["dow"] >= 5
    df["week_num"]   = df["date"].dt.isocalendar().week.astype(int)
    df["month"]      = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    return df

def detect_holidays(df: pd.DataFrame, min_run: int = 3) -> pd.DataFrame:
    """Flag consecutive weekend+day-off runs ≥ min_run as 'holiday'."""
    df = df.copy()
    df["holiday"] = False
    if "is_weekend" not in df.columns:
        return df
    dates = set(df["date"])
    for i, row in df.iterrows():
        if row["is_weekend"]:
            # look for surrounding block
            run = 0
            d = row["date"]
            while d in dates:
                run += 1
                d += timedelta(days=1)
            d = row["date"] - timedelta(days=1)
            while d in dates:
                run += 1
                d -= timedelta(days=1)
            if run >= min_run:
                df.at[i, "holiday"] = True
    return df

def safe_col(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None

# ── metric card html ───────────────────────────────────────────────────────────
def metric_card(label, value, delta="", color=ACCENT):
    return f"""
<div class="metric-card">
  <div class="accent-bar" style="background:{color}"></div>
  <div class="label">{label}</div>
  <div class="value">{value}</div>
  {"<div class='delta'>" + delta + "</div>" if delta else ""}
</div>"""

def insight_card(icon, tag_label, tag_class, text):
    return f"""
<div class="insight-card">
  <div class="insight-icon">{icon}</div>
  <div>
    <span class="insight-tag {tag_class}">{tag_label}</span>
    <div class="insight-text">{text}</div>
  </div>
</div>"""

# ── insights engine ─────────────────────────────────────────────────────────────
def generate_insights(df: pd.DataFrame) -> list[dict]:
    insights = []
    rec = safe_col(df, "recovery_score")
    hrv = safe_col(df, "hrv_rmssd_milli")
    rhr = safe_col(df, "resting_heart_rate")
    str_col = safe_col(df, "strain")
    slp = safe_col(df, "sleep_performance_pct")

    work = df[~df["is_weekend"]]
    wknd = df[df["is_weekend"]]

    # 1. Workday vs weekend recovery
    if rec:
        wd_rec = work[rec].mean()
        we_rec = wknd[rec].mean()
        diff = we_rec - wd_rec
        if abs(diff) > 3:
            direction = "higher" if diff > 0 else "lower"
            tag = "tag-info" if diff > 0 else "tag-warn"
            insights.append(dict(
                icon="◎", tag="pattern", tag_class=tag,
                text=f"Your recovery is <strong>{abs(diff):.0f} pts {direction} on weekends</strong> vs workdays "
                     f"({we_rec:.0f} vs {wd_rec:.0f}). "
                     + ("Weekends are genuinely restoring you." if diff > 0
                        else "Your weekends aren't recovering you as much as you'd expect — check sleep consistency.")
            ))

    # 2. Friday drag
    if rec:
        fri = df[df["dow"] == 4][rec].mean()
        mon = df[df["dow"] == 0][rec].mean()
        if fri < mon - 4:
            insights.append(dict(
                icon="📉", tag="weekly rhythm", tag_class="tag-warn",
                text=f"Friday recovery averages <strong>{fri:.0f}</strong> vs Monday's <strong>{mon:.0f}</strong>. "
                     f"The week is draining you by {mon - fri:.0f} pts. "
                     "Consider a lower-strain Thursday to protect the end of your week."
            ))

    # 3. Holiday signal
    if "holiday" in df.columns and rec:
        hol = df[df["holiday"] == True]
        nonhol = df[df["holiday"] == False]
        if len(hol) >= 3:
            hol_rec = hol[rec].mean()
            non_rec = nonhol[rec].mean()
            diff = hol_rec - non_rec
            insights.append(dict(
                icon="🌴", tag="holiday effect", tag_class="tag-good" if diff > 0 else "tag-warn",
                text=f"On holiday stretches your recovery is <strong>{hol_rec:.0f}</strong> "
                     f"vs {non_rec:.0f} otherwise — a <strong>{abs(diff):.0f} pt {'boost' if diff > 0 else 'drop'}</strong>. "
                     + ("Holidays are working. Your body actually loves them." if diff > 0
                        else "Your holidays might be active/travel ones — recovery isn't peaking the way rest should allow.")
            ))

    # 4. "Your body is screaming for a break" detector
    if rec:
        rolling = df[rec].rolling(7, min_periods=5).mean()
        latest_7 = rolling.iloc[-1]
        overall = df[rec].mean()
        if latest_7 < overall - 8:
            insights.append(dict(
                icon="🔴", tag="urgent", tag_class="tag-warn",
                text=f"Your 7-day average recovery is <strong>{latest_7:.0f}</strong> vs your baseline of "
                     f"<strong>{overall:.0f}</strong>. That's a <strong>{overall - latest_7:.0f} pt gap</strong>. "
                     "Your body is signalling for a break. A 2–3 day low-strain period could reset this significantly."
            ))

    # 5. HRV suppression on workdays
    if hrv:
        wd_hrv = work[hrv].mean()
        we_hrv = wknd[hrv].mean()
        if we_hrv > wd_hrv * 1.05:
            insights.append(dict(
                icon="💓", tag="hrv rhythm", tag_class="tag-info",
                text=f"HRV is <strong>{we_hrv:.0f} ms on weekends</strong> vs {wd_hrv:.0f} ms on workdays. "
                     "Stress (not exercise) is likely suppressing weekday HRV. "
                     "Even a 10-min walk at lunch has been shown to bump next-day HRV."
            ))

    # 6. Strain vs recovery mismatch
    if str_col and rec:
        high_strain_low_rec = df[(df[str_col] > df[str_col].quantile(.75)) & (df[rec] < 50)]
        if len(high_strain_low_rec) >= 5:
            insights.append(dict(
                icon="⚡", tag="overreach risk", tag_class="tag-warn",
                text=f"On <strong>{len(high_strain_low_rec)} days</strong> you pushed high strain "
                     f"(top 25%) while already in the red on recovery (<50). "
                     "These are your highest injury & burnout risk days. Check if they cluster on specific weekdays."
            ))

    # 7. Best sleep day
    if slp:
        best_dow = df.groupby("dow_name")[slp].mean().idxmax()
        worst_dow = df.groupby("dow_name")[slp].mean().idxmin()
        best_val = df.groupby("dow_name")[slp].mean().max()
        worst_val = df.groupby("dow_name")[slp].mean().min()
        insights.append(dict(
            icon="🌙", tag="sleep pattern", tag_class="tag-good",
            text=f"Your best sleep is on <strong>{best_dow}s ({best_val:.0f}%)</strong>, "
                 f"your worst on <strong>{worst_dow}s ({worst_val:.0f}%)</strong>. "
                 "Protecting your pre-sleep routine on "
                 + worst_dow + "s could meaningfully shift your weekly recovery arc."
        ))

    # 8. Respiratory rate trend
    if "respiratory_rate" in df.columns:
        recent = df.tail(14)["respiratory_rate"].mean()
        baseline = df["respiratory_rate"].mean()
        diff = recent - baseline
        if abs(diff) > 0.3:
            direction = "elevated" if diff > 0 else "lower"
            tag = "tag-warn" if diff > 0 else "tag-good"
            insights.append(dict(
                icon="🫁", tag="respiratory signal", tag_class=tag,
                text=f"Your respiratory rate is <strong>{direction} this fortnight</strong> "
                     f"({recent:.1f} vs {baseline:.1f} breaths/min baseline). "
                     + ("Elevated RR can precede illness or overtraining by 3–5 days." if diff > 0
                        else "Lower RR is a sign of deepening cardiovascular fitness.")
            ))

    return insights


# ── chart helpers ──────────────────────────────────────────────────────────────
def make_recovery_timeline(df, rec):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df[rec].rolling(7, min_periods=3).mean(),
        line=dict(color=ACCENT, width=2),
        name="7-day avg", hovertemplate="%{y:.0f}%<extra>7d avg</extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df[rec],
        line=dict(color=MUTED, width=1),
        mode="lines",
        name="daily", hovertemplate="%{y:.0f}%<extra>daily</extra>"
    ))
    fig.add_hrect(y0=67, y1=100, fillcolor=ACCENT, opacity=.04, line_width=0)
    fig.add_hrect(y0=34, y1=67, fillcolor=WARN, opacity=.04, line_width=0)
    fig.add_hrect(y0=0, y1=34, fillcolor=ACCENT3, opacity=.04, line_width=0)
    fig.update_layout(**PLOTLY_LAYOUT, height=220, showlegend=False)
    return fig

def make_weekday_bars(df, col, color=ACCENT):
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    grp = df.groupby("dow_name")[col].mean().reindex(order).dropna()
    colors = [color if d in ["Saturday","Sunday"] else MUTED for d in grp.index]
    fig = go.Figure(go.Bar(
        x=grp.index, y=grp.values,
        marker_color=colors,
        hovertemplate="%{x}: %{y:.1f}<extra></extra>"
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=200, bargap=.35)
    fig.update_xaxes(tickfont=dict(size=9))
    return fig

def make_scatter(df, x_col, y_col, color=ACCENT):
    fig = go.Figure(go.Scatter(
        x=df[x_col], y=df[y_col],
        mode="markers",
        marker=dict(color=color, size=5, opacity=.5),
        hovertemplate=f"{x_col}: %{{x:.1f}}<br>{y_col}: %{{y:.1f}}<extra></extra>"
    ))
    # trend line
    valid = df[[x_col, y_col]].dropna()
    if len(valid) > 5:
        m, b = np.polyfit(valid[x_col], valid[y_col], 1)
        xr = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)
        fig.add_trace(go.Scatter(x=xr, y=m*xr+b,
            line=dict(color=ACCENT2, width=1.5, dash="dot"),
            hoverinfo="skip", showlegend=False))
    fig.update_layout(**PLOTLY_LAYOUT, height=240)
    return fig

def make_monthly_heatmap(df, col):
    pivot = df.pivot_table(index="month_name", columns="dow_name", values=col, aggfunc="mean")
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex([m for m in month_order if m in pivot.index])
    pivot = pivot.reindex(columns=[d for d in dow_order if d in pivot.columns])
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0, "#1a0a1a"], [0.5, "#2a2a1a"], [1, ACCENT]],
        hovertemplate="%{y} %{x}: %{z:.0f}<extra></extra>",
        showscale=False,
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=260)
    fig.update_xaxes(tickfont=dict(size=9))
    return fig


# ── main app ───────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="hero">
      <div class="hero-title">Body<br><span class="hero-accent">Intelligence</span></div>
      <div class="hero-sub">◉ the whoop insights you actually wanted</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your Whoop CSV export here",
        type=["csv"],
        label_visibility="collapsed"
    )

    if uploaded is None:
        st.markdown("""
        <div class="upload-hero">
          <div class="upload-title">Drop your<br>Whoop data<span class="dot">.</span></div>
          <div class="upload-sub">
            Export from Whoop app → Profile → App Settings → Export Data.<br>
            Your data never leaves your browser.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # sample data for demo
        with st.expander("▸ try with sample data instead"):
            if st.button("Load 200-day sample", type="secondary"):
                st.session_state["use_sample"] = True
                st.rerun()
        return

    # ── load & enrich ──────────────────────────────────────────────────────────
    df = load_data(uploaded)
    df = detect_holidays(df)

    rec    = safe_col(df, "recovery_score")
    hrv    = safe_col(df, "hrv_rmssd_milli")
    rhr    = safe_col(df, "resting_heart_rate")
    strain = safe_col(df, "strain")
    slp    = safe_col(df, "sleep_performance_pct")
    slp_c  = safe_col(df, "sleep_consistency_pct")
    rem    = safe_col(df, "rem_sleep_mins")
    sws    = safe_col(df, "slow_wave_sleep_mins")

    days = len(df)
    date_range = f"{df['date'].min():%b %d} – {df['date'].max():%b %d, %Y}"

    # ── kpi row ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">overview · ' + date_range + f' · {days} days</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    kpis = []
    if rec:
        avg_rec = df[rec].mean()
        recent_rec = df.tail(14)[rec].mean()
        kpis.append((cols[0], "Avg Recovery", f"{avg_rec:.0f}%",
                     f"↑{recent_rec - avg_rec:+.0f} last 14d", ACCENT))
    if hrv:
        avg_hrv = df[hrv].mean()
        kpis.append((cols[1], "Avg HRV", f"{avg_hrv:.0f}ms", "", ACCENT2))
    if rhr:
        avg_rhr = df[rhr].mean()
        kpis.append((cols[2], "Resting HR", f"{avg_rhr:.0f}bpm", "", ACCENT3))
    if strain:
        avg_str = df[strain].mean()
        kpis.append((cols[3], "Avg Strain", f"{avg_str:.1f}", "", WARN))
    if slp:
        avg_slp = df[slp].mean()
        kpis.append((cols[4], "Sleep Score", f"{avg_slp:.0f}%", "", ACCENT2))

    for col, label, val, delta, color in kpis:
        col.markdown(metric_card(label, val, delta, color), unsafe_allow_html=True)

    # ── tabs ──────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["◎  insights", "◈  patterns", "◇  sleep", "◫  explore"])

    # ── TAB 1: insights ───────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">AI-generated insights from your data</div>', unsafe_allow_html=True)
        insights = generate_insights(df)
        if not insights:
            st.markdown("_Not enough data variation to surface insights yet. Try with more days._")
        for ins in insights:
            st.markdown(
                insight_card(ins["icon"], ins["tag"], ins["tag_class"], ins["text"]),
                unsafe_allow_html=True
            )

        # Recovery timeline
        if rec:
            st.markdown('<div class="section-header">recovery over time</div>', unsafe_allow_html=True)
            st.plotly_chart(make_recovery_timeline(df, rec), use_container_width=True, config={"displayModeBar": False})

    # ── TAB 2: patterns ───────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">workday vs weekend patterns</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        if rec:
            with c1:
                st.markdown("**Recovery by day of week**")
                st.plotly_chart(make_weekday_bars(df, rec, ACCENT), use_container_width=True, config={"displayModeBar": False})
        if hrv:
            with c2:
                st.markdown("**HRV by day of week**")
                st.plotly_chart(make_weekday_bars(df, hrv, ACCENT2), use_container_width=True, config={"displayModeBar": False})

        if strain and rec:
            st.markdown('<div class="section-header">strain vs recovery relationship</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Strain → next-day recovery**")
                df_lead = df.copy()
                df_lead["next_rec"] = df_lead[rec].shift(-1)
                st.plotly_chart(make_scatter(df_lead.dropna(subset=[strain, "next_rec"]), strain, "next_rec", WARN),
                    use_container_width=True, config={"displayModeBar": False})
            with c2:
                if hrv:
                    st.markdown("**HRV → recovery**")
                    st.plotly_chart(make_scatter(df.dropna(subset=[hrv, rec]), hrv, rec, ACCENT2),
                        use_container_width=True, config={"displayModeBar": False})

        # Holiday comparison
        if "holiday" in df.columns and rec:
            hol = df[df["holiday"] == True]
            nonhol = df[df["holiday"] == False]
            if len(hol) >= 3:
                st.markdown('<div class="section-header">holiday vs regular days</div>', unsafe_allow_html=True)
                fig = go.Figure()
                for subset, label, color in [(nonhol, "Regular", MUTED), (hol, "Holiday", ACCENT)]:
                    fig.add_trace(go.Box(
                        y=subset[rec], name=label,
                        marker_color=color,
                        line_color=color,
                        fillcolor=color.replace(")", ", 0.15)").replace("rgb", "rgba") if "rgb" in color else color + "26",
                        hovertemplate="%{y:.0f}%<extra>" + label + "</extra>",
                        boxmean=True,
                    ))
                fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── TAB 3: sleep ──────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">sleep architecture</div>', unsafe_allow_html=True)

        if rem and sws:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**REM sleep by weekday**")
                st.plotly_chart(make_weekday_bars(df, rem, ACCENT2), use_container_width=True, config={"displayModeBar": False})
            with c2:
                st.markdown("**Deep (SWS) sleep by weekday**")
                st.plotly_chart(make_weekday_bars(df, sws, ACCENT3), use_container_width=True, config={"displayModeBar": False})

        if slp:
            st.markdown('<div class="section-header">sleep score heatmap — month × weekday</div>', unsafe_allow_html=True)
            st.plotly_chart(make_monthly_heatmap(df, slp), use_container_width=True, config={"displayModeBar": False})

        if slp_c and slp:
            st.markdown('<div class="section-header">consistency vs performance</div>', unsafe_allow_html=True)
            st.plotly_chart(make_scatter(df.dropna(subset=[slp_c, slp]), slp_c, slp, ACCENT),
                use_container_width=True, config={"displayModeBar": False})

    # ── TAB 4: explore ────────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">build your own view</div>', unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["dow", "week_num", "month"]]

        c1, c2 = st.columns(2)
        x_col = c1.selectbox("X axis", numeric_cols, index=numeric_cols.index(strain) if strain in numeric_cols else 0)
        y_col = c2.selectbox("Y axis", numeric_cols, index=numeric_cols.index(rec) if rec in numeric_cols else 1)

        color_by = st.radio("Colour by", ["none", "weekend", "holiday"], horizontal=True)
        valid = df.dropna(subset=[x_col, y_col])

        if color_by == "weekend":
            colors = [ACCENT if w else MUTED for w in valid["is_weekend"]]
            labels = ["weekend" if w else "workday" for w in valid["is_weekend"]]
        elif color_by == "holiday" and "holiday" in valid.columns:
            colors = [ACCENT if h else MUTED for h in valid["holiday"]]
            labels = ["holiday" if h else "regular" for h in valid["holiday"]]
        else:
            colors = ACCENT
            labels = None

        fig = go.Figure(go.Scatter(
            x=valid[x_col], y=valid[y_col],
            mode="markers",
            marker=dict(color=colors, size=6, opacity=.6),
            text=labels,
            hovertemplate=f"{x_col}: %{{x:.1f}}<br>{y_col}: %{{y:.1f}}<extra>%{{text}}</extra>" if labels
                          else f"{x_col}: %{{x:.1f}}<br>{y_col}: %{{y:.1f}}<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=380)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # correlation table
        st.markdown('<div class="section-header">correlation with ' + y_col + '</div>', unsafe_allow_html=True)
        corr = df[numeric_cols].corr()[y_col].drop(y_col).sort_values(key=abs, ascending=False).head(8)
        corr_df = pd.DataFrame({"metric": corr.index, "r": corr.values.round(3)})
        corr_df["direction"] = corr_df["r"].apply(lambda x: "↑ positive" if x > 0 else "↓ negative")
        st.dataframe(
            corr_df,
            column_config={
                "r": st.column_config.ProgressColumn("correlation r", min_value=-1, max_value=1),
            },
            hide_index=True,
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
