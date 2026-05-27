import io
import html
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from services.bridge_ai import analyze_bridge_vibration
from services.demo_data import create_demo_dataframe, dataframe_to_csv_bytes
from services.history import add_bridge, add_history_record, ensure_state


st.set_page_config(
    page_title="桥梁振动检测 AI",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded",
)


@dataclass
class AccelerationData:
    time: np.ndarray
    acceleration: np.ndarray
    sample_rate: float
    source_column: str


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @keyframes fadeUp {
            from {
                opacity: 0;
                transform: translateY(18px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        @keyframes pulseLine {
            0%, 100% {
                opacity: 0.42;
                transform: scaleX(0.96);
            }
            50% {
                opacity: 0.9;
                transform: scaleX(1);
            }
        }
        @keyframes waveMove {
            from {
                background-position: 0 0, 0 0;
            }
            to {
                background-position: 760px 0, -520px 0;
            }
        }
        #MainMenu, footer, header {
            visibility: hidden;
        }
        .stDeployButton {
            display: none !important;
        }
        .stApp {
            background:
                radial-gradient(circle at 18% 10%, rgba(45, 212, 191, 0.20), transparent 28%),
                radial-gradient(circle at 85% 14%, rgba(99, 102, 241, 0.22), transparent 30%),
                radial-gradient(circle at 58% 86%, rgba(14, 165, 233, 0.14), transparent 34%),
                linear-gradient(145deg, #030712 0%, #09111f 42%, #050814 100%);
            color: #e5f3ff;
        }
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
                linear-gradient(rgba(148, 163, 184, 0.055) 1px, transparent 1px),
                linear-gradient(90deg, rgba(148, 163, 184, 0.045) 1px, transparent 1px);
            background-size: 54px 54px;
            mask-image: linear-gradient(to bottom, rgba(0,0,0,0.72), transparent 76%);
        }
        [data-testid="stSidebar"] {
            background: rgba(5, 10, 20, 0.72);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border-right: 1px solid rgba(148, 163, 184, 0.20);
        }
        [data-testid="stSidebar"] * {
            color: #dbeafe;
        }
        .block-container {
            max-width: 1380px;
            padding-top: 3.2rem;
            padding-bottom: 3.5rem;
            animation: fadeUp 680ms ease both;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        h2, h3, .stSubheader {
            color: #f8fbff;
        }
        div[data-testid="stVerticalBlock"] > div:has(> .stPlotlyChart),
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.68), rgba(15, 23, 42, 0.36));
            border-radius: 18px;
            padding: 12px;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.30);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
        }
        .hero {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(125, 211, 252, 0.22);
            background:
                linear-gradient(135deg, rgba(14, 165, 233, 0.24), rgba(79, 70, 229, 0.13) 46%, rgba(15, 23, 42, 0.60)),
                rgba(15, 23, 42, 0.42);
            border-radius: 26px;
            padding: 54px 58px;
            box-shadow: 0 32px 110px rgba(0, 0, 0, 0.42), inset 0 1px 0 rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            margin-bottom: 34px;
        }
        .hero::before {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            opacity: 0.24;
            background:
                repeating-radial-gradient(ellipse at 20% 80%, transparent 0 32px, rgba(103, 232, 249, 0.38) 33px 34px, transparent 35px 70px),
                repeating-radial-gradient(ellipse at 82% 20%, transparent 0 38px, rgba(196, 181, 253, 0.28) 39px 40px, transparent 41px 86px);
            animation: waveMove 18s linear infinite;
        }
        .hero::after {
            content: "";
            position: absolute;
            left: 44px;
            right: 44px;
            bottom: 28px;
            height: 2px;
            border-radius: 999px;
            background: linear-gradient(90deg, transparent, #22d3ee, #a78bfa, transparent);
            animation: pulseLine 3.2s ease-in-out infinite;
        }
        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: #67e8f9;
            font-size: 0.84rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 18px;
        }
        .logo-strip {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            margin-bottom: 22px;
        }
        .brand-logo {
            display: inline-flex;
            align-items: center;
            gap: 12px;
            color: #e0f2fe;
            font-weight: 820;
            letter-spacing: 0.02em;
        }
        .logo-mark {
            width: 42px;
            height: 42px;
            border-radius: 14px;
            display: inline-grid;
            place-items: center;
            background: linear-gradient(135deg, #22d3ee, #8b5cf6);
            color: #020617;
            box-shadow: 0 16px 50px rgba(34, 211, 238, 0.22);
            font-weight: 900;
        }
        .award-tag {
            border: 1px solid rgba(103, 232, 249, 0.26);
            border-radius: 999px;
            padding: 9px 14px;
            color: #b8d7ef;
            background: rgba(2, 6, 23, 0.26);
        }
        .hero-title {
            font-size: clamp(3.15rem, 6.2vw, 6.6rem);
            line-height: 0.94;
            font-weight: 820;
            margin: 0;
            color: #f8fbff;
            text-shadow: 0 18px 70px rgba(34, 211, 238, 0.18);
        }
        .hero-subtitle {
            margin-top: 24px;
            color: #b8c7dc;
            font-size: 1.12rem;
            line-height: 1.85;
            max-width: 900px;
        }
        .panel {
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.68), rgba(15, 23, 42, 0.42));
            border-radius: 22px;
            padding: 24px;
            box-shadow: 0 24px 90px rgba(0, 0, 0, 0.30);
            backdrop-filter: blur(22px);
            -webkit-backdrop-filter: blur(22px);
            margin: 18px 0 28px;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 18px;
            margin: 18px 0 34px;
        }
        .feature-card, .flow-card, .footer-panel {
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: linear-gradient(145deg, rgba(15, 23, 42, 0.72), rgba(30, 41, 59, 0.36));
            border-radius: 22px;
            padding: 24px;
            box-shadow: 0 20px 70px rgba(0, 0, 0, 0.24);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
        }
        .feature-card h3, .flow-card h3 {
            margin: 0 0 10px;
            color: #f8fbff;
            font-size: 1.05rem;
        }
        .feature-card p, .flow-card p, .footer-panel p {
            margin: 0;
            color: #aebfd3;
            line-height: 1.75;
        }
        .flow-row {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 14px;
            margin: 18px 0 34px;
        }
        .flow-step {
            position: relative;
            min-height: 132px;
            border: 1px solid rgba(103, 232, 249, 0.20);
            border-radius: 20px;
            padding: 20px;
            background: linear-gradient(145deg, rgba(8, 47, 73, 0.48), rgba(49, 46, 129, 0.24));
        }
        .flow-index {
            width: 34px;
            height: 34px;
            border-radius: 12px;
            display: grid;
            place-items: center;
            color: #020617;
            font-weight: 900;
            background: linear-gradient(135deg, #67e8f9, #c4b5fd);
            margin-bottom: 12px;
        }
        .flow-step strong {
            display: block;
            color: #f8fbff;
            margin-bottom: 8px;
        }
        .flow-step span {
            color: #aebfd3;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .metric-card {
            min-height: 132px;
            border: 1px solid rgba(125, 211, 252, 0.20);
            background:
                linear-gradient(145deg, rgba(34, 211, 238, 0.14), rgba(99, 102, 241, 0.08) 52%, rgba(15, 23, 42, 0.62)),
                rgba(15, 23, 42, 0.62);
            border-radius: 20px;
            padding: 22px;
            box-shadow: 0 18px 60px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
        }
        .metric-card:hover {
            transform: translateY(-3px);
            border-color: rgba(103, 232, 249, 0.46);
            box-shadow: 0 24px 74px rgba(8, 145, 178, 0.18);
        }
        .metric-label {
            color: #9fb4ca;
            font-size: 0.82rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .metric-value {
            background: linear-gradient(90deg, #67e8f9, #c4b5fd);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            font-size: clamp(1.55rem, 2.5vw, 2.4rem);
            font-weight: 820;
        }
        .hint {
            color: #9fb6c9;
            font-size: 0.95rem;
            margin: 14px 2px 28px;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #f8fbff;
            margin: 0 0 14px;
        }
        .ai-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 18px;
            margin: 14px 0 28px;
        }
        .ai-card {
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: linear-gradient(145deg, rgba(15, 23, 42, 0.76), rgba(30, 41, 59, 0.42));
            border-radius: 22px;
            padding: 24px;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
            backdrop-filter: blur(22px);
            -webkit-backdrop-filter: blur(22px);
            transition: transform 180ms ease, border-color 180ms ease;
        }
        .ai-card:hover {
            transform: translateY(-2px);
            border-color: rgba(125, 211, 252, 0.34);
        }
        .ai-card h3 {
            margin: 0 0 12px;
            font-size: 1.05rem;
        }
        .ai-card p {
            margin: 8px 0;
            color: #bfd0e4;
        }
        .ai-conclusion {
            margin-top: 12px;
            padding: 16px;
            border: 1px solid rgba(103, 232, 249, 0.20);
            border-radius: 16px;
            background: rgba(2, 6, 23, 0.30);
            color: #e2f3ff;
            line-height: 1.85;
        }
        .risk-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 88px;
            height: 34px;
            border-radius: 999px;
            color: #020617;
            font-weight: 800;
            margin-bottom: 10px;
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.26);
        }
        .meta-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin-top: 12px;
        }
        .meta-item {
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 16px;
            padding: 13px 14px;
            background: rgba(2, 6, 23, 0.28);
        }
        .meta-label {
            color: #8ea4bd;
            font-size: 0.78rem;
            margin-bottom: 4px;
        }
        .meta-value {
            color: #f8fbff;
            font-weight: 700;
            word-break: break-word;
        }
        div[data-testid="stFileUploader"] {
            border: 1px dashed rgba(103, 232, 249, 0.46);
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.12), rgba(15, 23, 42, 0.58));
            border-radius: 22px;
            padding: 22px;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
            transition: border-color 180ms ease, background 180ms ease;
        }
        div[data-testid="stFileUploader"]:hover {
            border-color: rgba(196, 181, 253, 0.66);
            background: linear-gradient(135deg, rgba(34, 211, 238, 0.16), rgba(79, 70, 229, 0.10));
        }
        .stAlert {
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.16);
            background: rgba(15, 23, 42, 0.62);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
        }
        [data-testid="stMarkdownContainer"] p {
            line-height: 1.75;
        }
        .stSelectbox, .stNumberInput, .stCheckbox {
            margin-bottom: 12px;
        }
        button, [data-testid="stBaseButton-secondary"] {
            border-radius: 999px !important;
        }
        @media (max-width: 780px) {
            .block-container {
                padding-top: 1.8rem;
            }
            .hero {
                border-radius: 22px;
                padding: 34px 24px 46px;
            }
            .hero-title {
                font-size: 3.1rem;
            }
            .ai-grid, .meta-grid {
                grid-template-columns: 1fr;
            }
            .feature-grid, .flow-row {
                grid-template-columns: 1fr;
            }
            .logo-strip {
                align-items: flex-start;
                flex-direction: column;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    return cleaned


def find_time_column(df: pd.DataFrame) -> str | None:
    keywords = ["time", "timestamp", "seconds", "second", "t", "时间", "时刻", "秒"]
    for col in df.columns:
        lower = str(col).lower()
        if any(key in lower for key in keywords):
            return col
    return None


def find_acceleration_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    keywords = [
        "acc",
        "acceleration",
        "accelerometer",
        "linear",
        "gforce",
        "x",
        "y",
        "z",
        "加速度",
        "加速",
    ]
    for col in df.columns:
        lower = str(col).lower()
        if any(key in lower for key in keywords) and pd.api.types.is_numeric_dtype(df[col]):
            candidates.append(col)
    if candidates:
        return candidates
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]


def read_csv(uploaded_file: io.BytesIO) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gbk", "latin1"]
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            uploaded_file.seek(0)
            return normalize_columns(pd.read_csv(uploaded_file, encoding=encoding))
        except Exception as exc:
            last_error = exc

    raise ValueError(f"无法读取 CSV 文件。最后一次错误：{last_error}")


def read_excel(uploaded_file: io.BytesIO) -> pd.DataFrame:
    try:
        uploaded_file.seek(0)
        excel_file = pd.ExcelFile(uploaded_file)
        if not excel_file.sheet_names:
            raise ValueError("Excel 文件里没有工作表。")
        return normalize_columns(pd.read_excel(excel_file, sheet_name=excel_file.sheet_names[0]))
    except ImportError as exc:
        raise ValueError("缺少 Excel 读取依赖，请先安装 openpyxl 和 xlrd。") from exc
    except Exception as exc:
        raise ValueError(f"无法读取 Excel 文件。请确认文件是 .xls 或 .xlsx 格式。错误：{exc}") from exc


def read_uploaded_file(uploaded_file: io.BytesIO) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return read_csv(uploaded_file)
    if file_name.endswith((".xls", ".xlsx")):
        return read_excel(uploaded_file)
    raise ValueError("暂时只支持 CSV、XLS、XLSX 文件。")


def build_acceleration_data(
    df: pd.DataFrame,
    selected_column: str | None,
    sample_rate_input: float,
    use_magnitude: bool,
) -> AccelerationData:
    if df.empty:
        raise ValueError("CSV 文件没有数据行。")

    time_col = find_time_column(df)
    acc_cols = find_acceleration_columns(df)

    if not acc_cols:
        raise ValueError("没有找到可用的数值型加速度列。请确认 CSV 中包含加速度数据。")

    working = df.copy()

    if time_col:
        working[time_col] = pd.to_numeric(working[time_col], errors="coerce")

    for col in acc_cols:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    if use_magnitude and len(acc_cols) >= 3:
        chosen_cols = acc_cols[:3]
        acceleration = np.sqrt(np.sum(np.square(working[chosen_cols].to_numpy(dtype=float)), axis=1))
        source_column = f"合加速度：{', '.join(chosen_cols)}"
    else:
        source_column = selected_column if selected_column in acc_cols else acc_cols[0]
        acceleration = working[source_column].to_numpy(dtype=float)

    if time_col:
        time = working[time_col].to_numpy(dtype=float)
        valid = np.isfinite(time) & np.isfinite(acceleration)
        time = time[valid]
        acceleration = acceleration[valid]
        order = np.argsort(time)
        time = time[order]
        acceleration = acceleration[order]
        diffs = np.diff(time)
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            raise ValueError("时间列无法计算采样间隔，请手动输入采样率。")
        sample_rate = 1.0 / float(np.median(diffs))
    else:
        acceleration = acceleration[np.isfinite(acceleration)]
        sample_rate = float(sample_rate_input)
        time = np.arange(len(acceleration), dtype=float) / sample_rate

    if len(acceleration) < 8:
        raise ValueError("有效数据点太少，至少需要 8 个点才能进行 FFT 分析。")

    if sample_rate <= 0:
        raise ValueError("采样率必须大于 0。")

    acceleration = acceleration - np.mean(acceleration)
    return AccelerationData(time=time, acceleration=acceleration, sample_rate=sample_rate, source_column=source_column)


def calculate_fft(data: AccelerationData) -> tuple[np.ndarray, np.ndarray, float, float]:
    n = len(data.acceleration)
    window = np.hanning(n)
    signal = data.acceleration * window

    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / data.sample_rate)
    amplitude = (2.0 / np.sum(window)) * np.abs(spectrum)

    valid = freqs > 0
    if not np.any(valid):
        raise ValueError("频谱中没有有效频率点。")

    valid_freqs = freqs[valid]
    valid_amp = amplitude[valid]
    peak_index = int(np.argmax(valid_amp))
    dominant_frequency = float(valid_freqs[peak_index])
    dominant_amplitude = float(valid_amp[peak_index])

    return freqs, amplitude, dominant_frequency, dominant_amplitude


def make_time_chart(data: AccelerationData) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.time,
            y=data.acceleration,
            mode="lines",
            line=dict(color="rgba(34, 211, 238, 0)", width=0),
            fill="tozeroy",
            fillcolor="rgba(34, 211, 238, 0.12)",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.time,
            y=data.acceleration,
            mode="lines",
            line=dict(color="#22d3ee", width=2.6, shape="spline", smoothing=0.45),
            name="加速度",
            hovertemplate="时间 %{x:.4f} s<br>加速度 %{y:.5f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="时域图",
        xaxis_title="时间 / s",
        yaxis_title="加速度（已去均值）",
        template="plotly_dark",
        height=460,
        margin=dict(l=28, r=24, t=62, b=32),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,6,23,0.34)",
        hovermode="x unified",
        font=dict(color="#dbeafe", family="Inter, Arial, sans-serif"),
        title_font=dict(size=22, color="#f8fbff"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.16)", rangeslider=dict(visible=True, thickness=0.08)),
        yaxis=dict(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.16)"),
    )
    return fig


def make_frequency_chart(freqs: np.ndarray, amplitude: np.ndarray, dominant_frequency: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=amplitude,
            mode="lines",
            line=dict(color="#a78bfa", width=2.5, shape="spline", smoothing=0.35),
            fill="tozeroy",
            fillcolor="rgba(167, 139, 250, 0.13)",
            name="FFT 幅值",
            hovertemplate="频率 %{x:.4f} Hz<br>幅值 %{y:.5f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=dominant_frequency,
        line_dash="dash",
        line_color="#facc15",
        line_width=2,
        annotation_text=f"主频 {dominant_frequency:.3f} Hz",
        annotation_position="top right",
    )
    fig.update_layout(
        title="频域图（FFT 频谱）",
        xaxis_title="频率 / Hz",
        yaxis_title="幅值",
        template="plotly_dark",
        height=460,
        margin=dict(l=28, r=24, t=62, b=32),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,6,23,0.34)",
        hovermode="x unified",
        font=dict(color="#dbeafe", family="Inter, Arial, sans-serif"),
        title_font=dict(size=22, color="#f8fbff"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.16)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.16)"),
    )
    return fig


def show_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_competition_home() -> None:
    st.markdown(
        """
        <section class="hero">
            <div class="logo-strip">
                <div class="brand-logo">
                    <span class="logo-mark">BI</span>
                    <span>BridgeSense AI</span>
                </div>
                <div class="award-tag">大学生创新创业大赛 · 科研转化项目</div>
            </div>
            <div class="hero-kicker">LIGHTWEIGHT BRIDGE HEALTH MONITORING</div>
            <p class="hero-title">基于智能手机传感器的桥梁轻量化监测系统</p>
            <p class="hero-subtitle">
                低成本、智能化、快速化农村桥梁健康监测平台。面向基层巡检场景，
                通过手机加速度传感器、FFT 频谱分析和 AI 工程判读，实现桥梁振动状态快速筛查。
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 项目介绍")
    st.markdown(
        """
        <div class="panel">
            本系统聚焦农村桥梁数量多、分布散、专业检测成本高的问题，
            将智能手机传感器采集、云端数据分析、AI 风险评估和 PDF 报告生成整合为轻量化平台。
            平台适合创新创业展示、科研训练、基层巡检辅助和结构健康监测原型验证。
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 技术亮点")
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <h3>低成本采集</h3>
                <p>利用智能手机加速度传感器完成初步振动数据采集，降低传统硬件部署门槛。</p>
            </div>
            <div class="feature-card">
                <h3>AI 工程判读</h3>
                <p>基于真实 FFT 主频、幅值、采样率和样本量生成专业检测结论。</p>
            </div>
            <div class="feature-card">
                <h3>报告闭环</h3>
                <p>自动输出时域图、频域图、风险等级和建议措施，形成可下载 PDF 报告。</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### AI检测流程图")
    st.markdown(
        """
        <div class="flow-row">
            <div class="flow-step"><div class="flow-index">1</div><strong>桥梁建档</strong><span>录入桥梁名称、类型、检测时间和检测人员。</span></div>
            <div class="flow-step"><div class="flow-index">2</div><strong>数据采集</strong><span>上传手机传感器导出的 CSV / XLS / XLSX 文件。</span></div>
            <div class="flow-step"><div class="flow-index">3</div><strong>频谱计算</strong><span>自动去均值、加窗并进行 FFT 频谱分析。</span></div>
            <div class="flow-step"><div class="flow-index">4</div><strong>AI 评估</strong><span>识别主频、幅值特征和潜在风险等级。</span></div>
            <div class="flow-step"><div class="flow-index">5</div><strong>报告导出</strong><span>生成工程检测风格 PDF 报告，便于归档与展示。</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer-panel">
            <p><strong>BridgeSense AI</strong> · 面向农村桥梁的轻量化智能监测平台</p>
            <p>本系统用于科研展示、教学实践和初步筛查。正式工程鉴定需结合专业检测规范、现场工况和长期监测数据。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_analysis(
    bridge_name: str,
    bridge_type: str,
    detection_time_text: str,
    inspector: str,
    analysis: dict,
    dominant_frequency: float,
    dominant_amplitude: float,
) -> None:
    safe_bridge_name = html.escape(bridge_name)
    safe_bridge_type = html.escape(bridge_type)
    safe_detection_time = html.escape(detection_time_text)
    safe_inspector = html.escape(inspector)
    risk_color = analysis["risk_color"]

    st.markdown(
        f"""
        <div class="ai-grid">
            <div class="ai-card">
                <h3>检测档案</h3>
                <div class="meta-grid">
                    <div class="meta-item">
                        <div class="meta-label">桥梁名称</div>
                        <div class="meta-value">{safe_bridge_name}</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">桥梁类型</div>
                        <div class="meta-value">{safe_bridge_type}</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">检测时间</div>
                        <div class="meta-value">{safe_detection_time}</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">检测人员</div>
                        <div class="meta-value">{safe_inspector}</div>
                    </div>
                </div>
            </div>
            <div class="ai-card">
                <h3>AI 分析结论</h3>
                <div class="risk-pill" style="background: {risk_color};">风险等级：{analysis["risk_level"]}</div>
                <p><strong>振动频率是否异常：</strong>{analysis["frequency_abnormal"]}</p>
                <p><strong>主频分析：</strong>{dominant_frequency:.3f} Hz，{analysis["frequency_note"]}</p>
                <p><strong>是否存在潜在风险：</strong>{analysis["potential_risk"]}</p>
                <p><strong>振动状态：</strong>{analysis["status"]}</p>
                <p><strong>建议措施：</strong>{analysis["advice"]}</p>
                <div class="ai-conclusion"><strong>工程检测结论：</strong>{analysis["engineering_conclusion"]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def register_pdf_fonts() -> str:
    font_name = "STSong-Light"
    try:
        pdfmetrics.registerFont(UnicodeCIDFont(font_name))
    except Exception:
        pass
    return font_name


def make_pdf_styles(font_name: str) -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName=font_name,
            fontSize=24,
            leading=32,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#EAF6FF"),
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontName=font_name,
            fontSize=10,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#9CCBEA"),
            spaceAfter=20,
        ),
        "section": ParagraphStyle(
            "ReportSection",
            parent=base["Heading2"],
            fontName=font_name,
            fontSize=15,
            leading=22,
            textColor=colors.HexColor("#54D8FF"),
            spaceBefore=10,
            spaceAfter=10,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base["Normal"],
            fontName=font_name,
            fontSize=10.5,
            leading=18,
            textColor=colors.HexColor("#D8E8F7"),
        ),
        "small": ParagraphStyle(
            "ReportSmall",
            parent=base["Normal"],
            fontName=font_name,
            fontSize=8.5,
            leading=13,
            textColor=colors.HexColor("#8FAAC1"),
        ),
    }


def add_pdf_background(canvas, doc) -> None:
    width, height = A4
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#050A16"))
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    canvas.setStrokeColor(colors.HexColor("#102A43"))
    canvas.setLineWidth(0.25)
    for x in range(0, int(width), 28):
        canvas.line(x, 0, x, height)
    for y in range(0, int(height), 28):
        canvas.line(0, y, width, y)
    canvas.setFillColor(colors.HexColor("#0E7490"))
    canvas.circle(26 * mm, height - 24 * mm, 16 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#312E81"))
    canvas.circle(width - 24 * mm, height - 28 * mm, 18 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#8FAAC1"))
    canvas.setFont("Helvetica", 7)
    canvas.drawRightString(width - 18 * mm, 11 * mm, f"Page {doc.page}")
    canvas.restoreState()


def chart_to_png_bytes(fig: go.Figure) -> io.BytesIO:
    image_bytes = fig.to_image(format="png", width=1300, height=760, scale=2)
    return io.BytesIO(image_bytes)


def risk_pdf_color(risk_level: str):
    if risk_level == "安全":
        return colors.HexColor("#22C55E")
    if risk_level == "注意":
        return colors.HexColor("#FACC15")
    return colors.HexColor("#EF4444")


def generate_pdf_report(
    bridge_name: str,
    bridge_type: str,
    detection_time_text: str,
    inspector: str,
    data: AccelerationData,
    freqs: np.ndarray,
    amplitude: np.ndarray,
    dominant_frequency: float,
    dominant_amplitude: float,
    analysis: dict,
) -> bytes:
    font_name = register_pdf_fonts()
    styles = make_pdf_styles(font_name)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title="桥梁振动检测AI报告",
    )

    story = [
        Paragraph("桥梁振动检测 AI 科研报告", styles["title"]),
        Paragraph("Bridge Vibration Detection & FFT Intelligent Analysis Report", styles["subtitle"]),
    ]

    info_rows = [
        ["桥梁名称", bridge_name, "桥梁类型", bridge_type],
        ["检测时间", detection_time_text, "检测人员", inspector],
        ["主频结果", f"{dominant_frequency:.3f} Hz", "主频幅值", f"{dominant_amplitude:.5f}"],
        ["采样率", f"{data.sample_rate:.2f} Hz", "有效点数", str(len(data.acceleration))],
        ["风险等级", analysis["risk_level"], "振动状态", analysis["status"]],
    ]
    info_table = Table(info_rows, colWidths=[28 * mm, 55 * mm, 28 * mm, 55 * mm])
    info_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 9.2),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0B1220")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#D8E8F7")),
                ("TEXTCOLOR", (0, 4), (1, 4), risk_pdf_color(analysis["risk_level"])),
                ("TEXTCOLOR", (2, 4), (3, 4), risk_pdf_color(analysis["risk_level"])),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#1E3A5F")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#0F2A44")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#0F2A44")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (1, 0), (1, -1), [colors.HexColor("#101A2E")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    story += [
        Paragraph("一、项目与桥梁信息", styles["section"]),
        info_table,
        Spacer(1, 10),
        Paragraph("二、AI 分析结论", styles["section"]),
        Paragraph(f"主频分析：{dominant_frequency:.3f} Hz，{analysis['frequency_note']}", styles["body"]),
        Paragraph(f"振动状态：{analysis['status']}。特征判断：{analysis['summary']}。", styles["body"]),
        Paragraph(f"风险等级：{analysis['risk_level']}。建议措施：{analysis['advice']}", styles["body"]),
        Paragraph("三、时域图", styles["section"]),
    ]

    time_png = chart_to_png_bytes(make_time_chart(data))
    time_image = Image(time_png, width=170 * mm, height=99 * mm)
    story += [time_image, Spacer(1, 8), Paragraph("图 1：加速度时域响应曲线", styles["small"]), PageBreak()]

    story += [
        Paragraph("四、频域图与主频识别", styles["section"]),
    ]
    freq_png = chart_to_png_bytes(make_frequency_chart(freqs, amplitude, dominant_frequency))
    freq_image = Image(freq_png, width=170 * mm, height=99 * mm)
    story += [
        freq_image,
        Spacer(1, 8),
        Paragraph("图 2：FFT 频谱图与主频标记", styles["small"]),
        Spacer(1, 14),
        Paragraph("五、工程建议", styles["section"]),
        Paragraph(analysis["advice"], styles["body"]),
        Spacer(1, 10),
        Paragraph(
            "说明：本报告由桥梁振动检测 AI 网页系统自动生成，适用于教学、科研展示和初步筛查。正式工程鉴定需结合结构图纸、现场工况、长期监测数据和专业检测规范。",
            styles["small"],
        ),
    ]

    doc.build(story, onFirstPage=add_pdf_background, onLaterPages=add_pdf_background)
    return buffer.getvalue()


def classify_risk(dominant_frequency: float, dominant_amplitude: float, sample_rate: float, point_count: int, bridge_type: str) -> dict:
    return analyze_bridge_vibration(
        dominant_frequency=dominant_frequency,
        dominant_amplitude=dominant_amplitude,
        sample_rate=sample_rate,
        point_count=point_count,
        bridge_type=bridge_type,
    )


def format_detection_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    inject_styles()
    ensure_state(st.session_state)
    demo_df = create_demo_dataframe()
    render_competition_home()

    st.sidebar.header("分析设置")
    sample_rate_input = st.sidebar.number_input(
        "没有时间列时使用的采样率（Hz）",
        min_value=1.0,
        max_value=1000.0,
        value=100.0,
        step=1.0,
        help="如果 CSV 中没有 time/时间 列，系统会使用这里的采样率生成时间轴。",
    )
    use_magnitude = st.sidebar.checkbox(
        "优先使用前三个加速度列计算合加速度",
        value=True,
        help="适合包含 x/y/z 三轴加速度的手机传感器数据。",
    )
    demo_mode = st.sidebar.toggle("一键演示模式", value=False, help="使用内置示例数据快速展示完整检测流程。")

    st.markdown("## 检测工作台")
    st.subheader("桥梁信息")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    add_bridge(st.session_state, "示例桥梁")
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        bridge_name = st.selectbox("选择桥梁", st.session_state["bridges"])
        new_bridge_name = st.text_input("新增桥梁名称", placeholder="例如：北河 2 号桥")
        if st.button("添加到桥梁库", use_container_width=True):
            add_bridge(st.session_state, new_bridge_name)
            st.rerun()
        bridge_type = st.selectbox("桥梁类型", ["梁桥", "连续梁桥", "拱桥", "斜拉桥", "悬索桥", "其他"])
    with info_col2:
        detection_date = st.date_input("检测日期")
        detection_time = st.time_input("检测时间")
        inspector = st.text_input("检测人员", value="检测员")
    detection_time_text = f"{detection_date} {detection_time}"
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("上传区")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.download_button(
        "下载示例数据",
        data=dataframe_to_csv_bytes(demo_df),
        file_name="bridge_demo_acceleration.csv",
        mime="text/csv",
        use_container_width=True,
    )
    uploaded_file = st.file_uploader("上传 CSV / Excel 文件", type=["csv", "xls", "xlsx"])
    if demo_mode:
        st.success("一键演示模式已启用：系统将使用内置示例数据完成完整流程。")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is None and not demo_mode:
        st.info("请上传手机加速度传感器导出的 CSV 文件。常见列名如 time、timestamp、acc_x、acc_y、acc_z。")
        with st.expander("查看示例 CSV 格式"):
            st.code(
                """time,acc_x,acc_y,acc_z
0.00,0.02,0.01,9.81
0.01,0.04,0.03,9.84
0.02,0.01,0.02,9.79""",
                language="csv",
            )
        render_footer()
        return

    try:
        if demo_mode:
            df = demo_df
            st.success("示例数据已载入。")
        else:
            df = read_uploaded_file(uploaded_file)
            st.success("文件读取成功。")

        st.subheader("数据预览")
        st.dataframe(df.head(20), use_container_width=True)

        acc_cols = find_acceleration_columns(df)
        selected_column = None
        if acc_cols:
            selected_column = st.selectbox("选择用于分析的加速度列", acc_cols)

        with st.spinner("AI分析中..."):
            data = build_acceleration_data(df, selected_column, sample_rate_input, use_magnitude)
            freqs, amplitude, dominant_frequency, dominant_amplitude = calculate_fft(data)
            analysis = classify_risk(
                dominant_frequency=dominant_frequency,
                dominant_amplitude=dominant_amplitude,
                sample_rate=data.sample_rate,
                point_count=len(data.acceleration),
                bridge_type=bridge_type,
            )
        add_history_record(
            st.session_state,
            bridge_name=bridge_name,
            bridge_type=bridge_type,
            dominant_frequency=dominant_frequency,
            risk_level=analysis["risk_level"],
            status=analysis["status"],
            point_count=len(data.acceleration),
        )

        st.subheader("结果区")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            show_metric_card("主频", f"{dominant_frequency:.3f} Hz")
        with col2:
            show_metric_card("主频幅值", f"{dominant_amplitude:.4f}")
        with col3:
            show_metric_card("采样率", f"{data.sample_rate:.2f} Hz")
        with col4:
            show_metric_card("有效点数", f"{len(data.acceleration)}")

        st.markdown(f'<p class="hint">当前分析数据源：{data.source_column}</p>', unsafe_allow_html=True)

        st.subheader("AI分析结果")
        render_ai_analysis(
            bridge_name=bridge_name,
            bridge_type=bridge_type,
            detection_time_text=detection_time_text,
            inspector=inspector,
            analysis=analysis,
            dominant_frequency=dominant_frequency,
            dominant_amplitude=dominant_amplitude,
        )

        st.subheader("桥梁检测报告导出系统")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<p class="hint">生成包含桥梁信息、AI分析结论、风险等级、时域图、频域图和建议措施的专业 PDF 报告。</p>',
            unsafe_allow_html=True,
        )
        try:
            with st.spinner("正在生成桥梁检测PDF报告..."):
                pdf_bytes = generate_pdf_report(
                    bridge_name=bridge_name,
                    bridge_type=bridge_type,
                    detection_time_text=detection_time_text,
                    inspector=inspector,
                    data=data,
                    freqs=freqs,
                    amplitude=amplitude,
                    dominant_frequency=dominant_frequency,
                    dominant_amplitude=dominant_amplitude,
                    analysis=analysis,
                )
            st.download_button(
                label="导出PDF报告",
                data=pdf_bytes,
                file_name=f"{bridge_name}_桥梁振动检测报告.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.caption(f"PDF报告已就绪，文件大小：{len(pdf_bytes) / 1024:.1f} KB")
        except Exception as pdf_exc:
            st.error("PDF报告生成失败，请确认 reportlab 和 kaleido 已安装。")
            st.warning(str(pdf_exc))
        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("图表区")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.plotly_chart(make_time_chart(data), use_container_width=True)
        with chart_col2:
            st.plotly_chart(make_frequency_chart(freqs, amplitude, dominant_frequency), use_container_width=True)

        with st.expander("分析说明"):
            st.write(
                "系统会先对加速度数据去均值，然后使用汉宁窗降低频谱泄漏，再通过 FFT 查找非零频率中幅值最大的点作为主频。"
            )

        st.subheader("数据历史记录")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        if st.session_state["history"]:
            st.dataframe(pd.DataFrame(st.session_state["history"]), use_container_width=True, hide_index=True)
        else:
            st.caption("暂无历史记录。完成一次检测后会自动记录。")
        st.markdown("</div>", unsafe_allow_html=True)
        render_footer()
    except Exception as exc:
        st.error("CSV 格式或数据内容无法完成分析。")
        st.warning(str(exc))
        st.markdown(
            """
            请检查：
            - 文件是否为 CSV；
            - 是否包含至少一个数值型加速度列；
            - 如果没有时间列，请在左侧输入正确采样率；
            - 有效数据点是否不少于 8 个。
            """
        )
        render_footer()


if __name__ == "__main__":
    main()
