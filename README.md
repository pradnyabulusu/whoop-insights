# Body Intelligence · Whoop Insights Dashboard

> *The insights Whoop doesn't tell you — but your body already knows.*

**[→ Live demo](https://your-app.streamlit.app)** &nbsp;|&nbsp; Built with Python + Streamlit + Plotly

---

## What this is

Whoop gives you a recovery score. What it doesn't give you is *context* — why is Wednesday always your worst day? Do holidays actually help you recover? Is your body quietly screaming for a break right now?

This dashboard takes your raw Whoop CSV export (200+ days of mine) and answers the questions the app never asks.

---

## Insights it surfaces

| Insight | What it catches |
|---|---|
| **Workday vs weekend recovery** | Whether work is measurably draining you |
| **Friday drag** | Cumulative weekly fatigue — your Monday vs Friday gap |
| **Holiday effect** | Do multi-day breaks actually show up in your HRV & recovery? |
| **"Take a break" detector** | 7-day rolling average falling below your baseline by >8 pts → explicit nudge |
| **HRV suppression** | Stress (not exercise) suppressing weekday HRV vs weekends |
| **Overreach risk days** | High strain on already-low recovery days — where burnout hides |
| **Best/worst sleep day** | Which weekday is consistently robbing your sleep |
| **Respiratory rate trend** | Elevated RR can precede illness by 3–5 days |

---

## How to run locally

```bash
git clone https://github.com/yourusername/whoop-insights
cd whoop-insights
pip install -r requirements.txt
streamlit run app.py
```

Then export your data from Whoop: **Profile → App Settings → Export Data**, and drop the CSV into the dashboard.

---

## How to deploy (free, 10 minutes)

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → New app
3. Point it at `app.py` — done

---

## Data columns used

The app auto-detects whichever of these are present in your export:

`recovery_score` · `hrv_rmssd_milli` · `resting_heart_rate` · `strain` · `sleep_performance_pct` · `sleep_consistency_pct` · `rem_sleep_mins` · `slow_wave_sleep_mins` · `respiratory_rate` · `skin_temp_celsius` · `spo2_percentage`

**Your data never leaves your machine** — the app runs entirely locally / in your browser session.

---

## Why I built this

Whoop is great hardware. The app insight layer always felt a bit... surface level. I wanted answers that sound like a coach who actually knows me:

- *"Your body's been in the red for 10 days — just go on that holiday"*
- *"Workdays drop your HRV by 12ms. That's not fitness, that's stress."*
- *"You sleep best on Tuesdays. Protect Tuesday nights."*

The data was always there. It just needed asking.

---

## Stack

- **Python** · data wrangling & insight logic
- **Streamlit** · UI & deployment
- **Plotly** · interactive charts
- **Pandas / NumPy** · analysis

---

*Built as part of a broader project exploring personal health data analysis.*
