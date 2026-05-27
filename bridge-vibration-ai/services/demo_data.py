from __future__ import annotations

import io

import numpy as np
import pandas as pd


def create_demo_dataframe(sample_rate: float = 100.0, duration: float = 12.0) -> pd.DataFrame:
    """Create deterministic bridge-like acceleration data for demo mode."""
    rng = np.random.default_rng(2026)
    time = np.arange(0, duration, 1 / sample_rate)
    acc_x = 0.055 * np.sin(2 * np.pi * 2.35 * time) + 0.010 * rng.normal(size=len(time))
    acc_y = 0.038 * np.sin(2 * np.pi * 2.35 * time + 0.55) + 0.008 * rng.normal(size=len(time))
    acc_z = 9.81 + 0.070 * np.sin(2 * np.pi * 2.35 * time + 0.2) + 0.012 * rng.normal(size=len(time))
    return pd.DataFrame(
        {
            "time": time.round(4),
            "acc_x": acc_x.round(6),
            "acc_y": acc_y.round(6),
            "acc_z": acc_z.round(6),
        }
    )


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")
