from __future__ import annotations

import numpy as np
from scipy import signal


def bandpass_filter(
    acceleration: np.ndarray,
    sample_rate: float,
    lowcut: float = 0.2,
    highcut: float = 20.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a bridge-vibration bandpass filter when sampling conditions allow it."""
    nyquist = sample_rate / 2.0
    safe_highcut = min(highcut, nyquist * 0.85)
    if safe_highcut <= lowcut or len(acceleration) < order * 12:
        return acceleration

    sos = signal.butter(order, [lowcut, safe_highcut], btype="bandpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, acceleration)


def calculate_welch_spectrum(
    acceleration: np.ndarray,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return Welch spectrum and dominant frequency from real acceleration data."""
    nperseg = min(1024, max(64, len(acceleration) // 4))
    freqs, power = signal.welch(
        acceleration,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="spectrum",
    )
    amplitude = np.sqrt(np.maximum(power, 0))
    valid = freqs > 0
    if not np.any(valid):
        raise ValueError("Welch 频谱中没有有效频率点。")

    valid_freqs = freqs[valid]
    valid_amp = amplitude[valid]
    peak_index = int(np.argmax(valid_amp))
    return freqs, amplitude, float(valid_freqs[peak_index]), float(valid_amp[peak_index])


def evaluate_signal_quality(
    acceleration: np.ndarray,
    sample_rate: float,
    dominant_amplitude: float,
    spectrum_amplitude: np.ndarray,
) -> dict:
    """Score data quality for engineering interpretation."""
    score = 100
    notes: list[str] = []
    valid_count = int(np.isfinite(acceleration).sum())

    if valid_count < 256:
        score -= 25
        notes.append("有效样本偏少")
    elif valid_count < 1024:
        score -= 10
        notes.append("样本量一般")
    else:
        notes.append("样本量充足")

    if sample_rate < 40:
        score -= 15
        notes.append("采样率偏低")
    else:
        notes.append("采样率满足初步频谱分析")

    std = float(np.std(acceleration))
    if std < 1e-5:
        score -= 30
        notes.append("信号波动过小，可能未采集到有效振动")

    if len(spectrum_amplitude) > 4:
        sorted_amp = np.sort(spectrum_amplitude[1:])
        background = float(np.median(sorted_amp)) + 1e-12
        peak_ratio = dominant_amplitude / background
        if peak_ratio < 3:
            score -= 20
            notes.append("主峰不够突出")
        else:
            notes.append("频谱主峰较清晰")
    else:
        peak_ratio = 0.0
        score -= 15
        notes.append("频谱点数偏少")

    score = max(0, min(100, score))
    level = "优秀" if score >= 85 else "良好" if score >= 70 else "一般" if score >= 55 else "较差"
    return {
        "score": score,
        "level": level,
        "notes": "；".join(notes),
        "peak_ratio": round(float(peak_ratio), 2),
    }


def compare_with_baseline(history: list[dict], bridge_name: str, current_frequency: float) -> dict:
    """Compare current dominant frequency with the latest historical record of the same bridge."""
    previous = [
        item for item in history
        if item.get("桥梁名称") == bridge_name and isinstance(item.get("主频(Hz)"), (int, float))
    ]
    if not previous:
        return {
            "has_baseline": False,
            "message": "暂无同桥梁历史基线，本次结果将作为后续对比参考。",
            "change_rate": None,
        }

    baseline_frequency = float(previous[0]["主频(Hz)"])
    if baseline_frequency <= 0:
        return {
            "has_baseline": False,
            "message": "历史基线无效，无法进行主频漂移对比。",
            "change_rate": None,
        }

    change_rate = abs(current_frequency - baseline_frequency) / baseline_frequency * 100
    if change_rate < 5:
        message = f"较历史基线变化 {change_rate:.2f}%，主频稳定。"
    elif change_rate < 10:
        message = f"较历史基线变化 {change_rate:.2f}%，建议持续关注。"
    else:
        message = f"较历史基线变化 {change_rate:.2f}%，存在明显漂移，建议复测并检查结构状态。"

    return {
        "has_baseline": True,
        "message": message,
        "change_rate": round(change_rate, 2),
        "baseline_frequency": round(baseline_frequency, 4),
    }
