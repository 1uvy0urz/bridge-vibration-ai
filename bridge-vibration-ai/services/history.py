from __future__ import annotations

from datetime import datetime


def ensure_state(state) -> None:
    state.setdefault("bridges", ["示例桥梁", "乡村东桥", "河道便桥"])
    state.setdefault("history", [])
    state.setdefault("last_record_key", "")


def add_bridge(state, bridge_name: str) -> None:
    name = bridge_name.strip()
    if name and name not in state["bridges"]:
        state["bridges"].append(name)


def add_history_record(
    state,
    bridge_name: str,
    bridge_type: str,
    dominant_frequency: float,
    risk_level: str,
    status: str,
    point_count: int,
) -> None:
    record_key = f"{bridge_name}-{dominant_frequency:.4f}-{risk_level}-{point_count}"
    if state.get("last_record_key") == record_key:
        return
    state["last_record_key"] = record_key
    state["history"].insert(
        0,
        {
            "检测时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "桥梁名称": bridge_name,
            "桥梁类型": bridge_type,
            "主频(Hz)": round(dominant_frequency, 4),
            "风险等级": risk_level,
            "状态": status,
            "有效点数": point_count,
        },
    )
    state["history"] = state["history"][:20]
