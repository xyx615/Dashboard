import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import re

st.set_page_config(page_title="Left Jab Dashboard", layout="wide")
G = 9.80665  # m/s^2 per g

# ---------- helpers ----------
def clean_and_derive(df: pd.DataFrame, roll_k: int) -> pd.DataFrame:
    rename_map = {
        "athlete":"athlete",
        "session":"session",
        "Punch #":"punch_index",
        "punch_index":"punch_index",
        "Time (s)":"t_peak",
        "t_peak":"t_peak",
        "Peak Accel (g)":"accel_peak_pos_g",
        "Peak Force (N)":"force_peak_N",
        "Cadence (s)":"cadence_interval_s",
        "Retraction Time (s)":"retraction_time_s",
        "Retraction Accel (g)":"retraction_accel_g",
        "Video Time (mm:ss.mmm)":"video_time_label",
    }
    df = df.rename(columns=rename_map)

    needed = ["t_peak","force_peak_N","accel_peak_pos_g"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    for c in ["punch_index","t_peak","force_peak_N","accel_peak_pos_g",
              "cadence_interval_s","retraction_time_s","retraction_accel_g"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # derive
    df["accel_peak_pos"] = df["accel_peak_pos_g"] * G
    if "retraction_accel_g" in df.columns:
        df["retraction_accel"] = df["retraction_accel_g"] * G
    if "cadence_interval_s" in df.columns:
        df["cadence_hz"] = 1.0 / df["cadence_interval_s"]
        df.loc[~np.isfinite(df["cadence_hz"]), "cadence_hz"] = np.nan

    if "athlete" not in df.columns:
        df["athlete"] = "Unknown"
    if "session" not in df.columns:
        df["session"] = "Session-1"

    df = df.sort_values(["athlete","session","t_peak"]).reset_index(drop=True)
    df["punch_index"] = df.groupby(["athlete","session"]).cumcount() + 1

    df["force_rm"] = df.groupby(["athlete","session"])["force_peak_N"]\
                       .transform(lambda s: s.rolling(roll_k, min_periods=3).mean())
    return df

def infer_athlete_from_name(stem: str) -> str:
    m = re.match(r"([A-Za-z]+)", stem)
    return m.group(1) if m else "Unknown"

def load_many_separately(files, roll_k: int) -> dict:
    out = {}
    for f in files:
        if f.name.lower().endswith(".csv"):
            tmp = pd.read_csv(f)
        else:
            tmp = pd.read_excel(f, sheet_name=0)
        if "athlete" not in tmp.columns:
            tmp["athlete"] = infer_athlete_from_name(Path(f.name).stem)
        if "session" not in tmp.columns:
            tmp["session"] = Path(f.name).stem
        out[f.name] = clean_and_derive(tmp, roll_k)
    return out

def coach_note(txt): st.caption(f"**Coach Note:** {txt}")

# ---------- sidebar: uploads & settings ----------
st.sidebar.header("Upload per-punch files (CSV/XLSX)")
files = st.sidebar.file_uploader(
    "Upload one or more files. Analysis runs per selected file.",
    type=["csv","xlsx","xls"],
    accept_multiple_files=True
)

roll_k = int(st.sidebar.number_input("Rolling window (force; punches)", 3, 50, 10, step=1))

if not files:
    st.info("Upload one or more per-punch files to begin.")
    st.stop()

# load individually
try:
    datasets = load_many_separately(files, roll_k)
except Exception as e:
    st.error(f"File error: {e}")
    st.stop()

# pick one file/session to view
file_key = st.sidebar.selectbox("Select file/session", list(datasets.keys()))
df = datasets[file_key].copy()

# optional athlete/session filter within the selected file
athletes = sorted(df["athlete"].unique())
ath_sel = st.sidebar.multiselect("Athletes in this file", athletes, default=athletes)
df = df[df["athlete"].isin(ath_sel)]

sessions = sorted(df["session"].unique())
ses_sel = st.sidebar.multiselect("Sessions in this file", sessions, default=sessions)
df = df[df["session"].isin(ses_sel)]

# time & punch filters
st.sidebar.markdown("---")
tmin, tmax = float(df["t_peak"].min()), float(df["t_peak"].max())
sel_t = st.sidebar.slider(
    "Time range (seconds)",
    min_value=float(np.floor(tmin)),
    max_value=float(np.ceil(tmax)),
    value=(float(np.floor(tmin)), float(np.ceil(tmax))),
    step=0.1
)
df = df[(df["t_peak"] >= sel_t[0]) & (df["t_peak"] <= sel_t[1])]

pmin = int(df["punch_index"].min())
pmax = int(df["punch_index"].max())
sel_p = st.sidebar.slider(
    "Punch number range",
    min_value=pmin,
    max_value=pmax,
    value=(pmin, pmax),
    step=1
)
df = df[(df["punch_index"] >= sel_p[0]) & (df["punch_index"] <= sel_p[1])]

# ---------- header ----------
st.title("Left Jab Coach Dashboard (single-session view)")
st.caption(f"Viewing: **{file_key}**")

# ---------- KPIs ----------
def kpi(val, label, fmt="{:.2f}"):
    st.markdown(f"**{label}**")
    if val is None or (isinstance(val, float) and np.isnan(val)):
        st.markdown("-")
    else:
        st.markdown(fmt.format(val))

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: kpi(df.shape[0], "Total Jabs", fmt="{:.0f}")
with c2: kpi(df["force_peak_N"].mean(), "Avg Peak Force (N)")
with c3: kpi(df["accel_peak_pos_g"].max(), "Max Peak Accel (g)")
with c4: kpi(df["cadence_hz"].mean(), "Avg Cadence (Hz)")
with c5: kpi(df["retraction_time_s"].mean(), "Mean Retraction Time (s)")
with c6:
    pct = 100*df["retraction_time_s"].notna().mean()
    kpi(pct, "Retraction Data Availability", fmt="{:.1f}%")
st.divider()

# ---------- Row 1 – Power / Speed ----------
col1, col2 = st.columns(2)

with col1:
    x_mode = st.radio("Force X-axis", ["Time (seconds)", "Punch index"], horizontal=True)
    xcol = "t_peak" if x_mode == "Time (seconds)" else "punch_index"
    xlab = "Time (seconds)" if xcol == "t_peak" else "Punch #"

    fig_force = go.Figure()
    for (a, s), g in df.groupby(["athlete","session"]):
        fig_force.add_trace(go.Scatter(
            x=g[xcol], y=g["force_peak_N"], mode="lines+markers",
            name=f"{a}-{s}", opacity=0.65
        ))
        fig_force.add_trace(go.Scatter(
            x=g[xcol], y=g["force_rm"], mode="lines",
            name=f"{a}-{s} (rolling)", line=dict(width=3)
        ))
    fig_force.update_layout(
        title="Force Over Time",
        xaxis_title=xlab, yaxis_title="Peak Force (N)", legend_title="Series"
    )
    st.plotly_chart(fig_force, use_container_width=True)
    coach_note("Use rolling trend to assess power drop or buildup across the selected punches/time window.")

with col2:
    color_by = st.selectbox("Color by", ["athlete", "session", "None"], index=0)
    color_col = None if color_by == "None" else color_by
    fig_acc = px.scatter(
        df.sort_values("t_peak"),
        x="t_peak", y="accel_peak_pos_g",
        color=color_col,
        hover_data=["athlete","session","punch_index","force_peak_N","retraction_time_s"],
        title="Peak Acceleration Over Time",
        labels={"t_peak":"Time (seconds)","accel_peak_pos_g":"Peak Accel (g)"}
    )
    st.plotly_chart(fig_acc, use_container_width=True)

# ---------- Row 2 – Retraction ----------
col3, col4 = st.columns(2)

with col3:
    target_min = st.number_input("Retraction target (min, s)", 0.10, 1.0, 0.30, 0.01)
    target_max = st.number_input("Retraction target (max, s)", 0.15, 1.5, 0.40, 0.01)
    fig_ret = go.Figure()
    for (a, s), g in df.groupby(["athlete","session"]):
        fig_ret.add_trace(go.Scatter(
            x=g["punch_index"], y=g["retraction_time_s"], mode="lines+markers",
            name=f"{a}-{s}", opacity=0.75
        ))
    if len(df):
        fig_ret.add_hrect(y0=target_min, y1=target_max, fillcolor="green", opacity=0.15, line_width=0)
    fig_ret.update_layout(
        title="Retraction Time Across Punches",
        xaxis_title="Punch #", yaxis_title="Retraction Time (s)"
    )
    st.plotly_chart(fig_ret, use_container_width=True)

with col4:
    if "retraction_accel_g" in df.columns:
        df_rt = df.dropna(subset=["retraction_accel_g"]).copy()
        fig_racc = px.scatter(
            df_rt, x="punch_index", y="retraction_accel_g",
            color="athlete",
            hover_data=["session","retraction_time_s","force_peak_N"],
            title="Retraction Acceleration Trends (g)",
            labels={"punch_index":"Punch #","retraction_accel_g":"Retraction Accel (g)"}
        )
        st.plotly_chart(fig_racc, use_container_width=True)
    else:
        st.info("`retraction_accel_g` not available in this file.")

# ---------- Row 3 – Distributions & Cadence ----------
col5, col6 = st.columns(2)

with col5:
    metric = st.selectbox("Distribution metric", ["force_peak_N","accel_peak_pos_g","retraction_time_s"], index=0)
    fig_hist = px.histogram(
        df, x=metric, color="athlete",
        nbins=40, marginal="box",
        title=f"Performance Distribution — {metric}",
        labels={metric:metric}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col6:
    fig_cad = px.line(
        df.sort_values("t_peak"),
        x="t_peak", y="cadence_hz", color="athlete",
        title="Cadence Over Time (per-punch Hz)",
        labels={"t_peak":"Time (seconds)","cadence_hz":"Cadence (Hz)"}
    )
    st.plotly_chart(fig_cad, use_container_width=True)

# ---------- table ----------
with st.expander("Data preview"):
    st.dataframe(df.head(200), use_container_width=True)

st.success("Viewing single-session data. Use the **file/session selector**, **time range**, and **punch range** filters on the left.")