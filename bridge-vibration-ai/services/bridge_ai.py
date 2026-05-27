from __future__ import annotations


def analyze_bridge_vibration(
    dominant_frequency: float,
    dominant_amplitude: float,
    sample_rate: float,
    point_count: int,
    bridge_type: str,
) -> dict:
    """Generate an engineering-style AI conclusion from real FFT results."""
    score = 0
    reasons: list[str] = []

    if dominant_amplitude >= 0.35:
        score += 2
        reasons.append("主频幅值偏高，说明当前振动响应较强")
    elif dominant_amplitude >= 0.18:
        score += 1
        reasons.append("主频幅值处于中等水平，建议结合历史数据复核")
    else:
        reasons.append("主频幅值较低，未见明显强振动响应")

    if dominant_frequency < 0.2:
        score += 1
        frequency_status = "主频偏低，可能与慢速整体摆动、采样时长或低频漂移有关"
    elif 0.2 <= dominant_frequency < 1.5:
        frequency_status = "主频处于桥梁低频响应范围，整体振动特征相对平稳"
    elif 1.5 <= dominant_frequency <= 8.0:
        score += 1
        frequency_status = "主频位于中低频敏感区，需要关注车辆、风荷载或局部构件响应影响"
    else:
        score += 1
        frequency_status = "主频偏高，可能受到局部扰动、传感器噪声或短时冲击影响"

    if point_count < 256:
        score += 1
        reasons.append("有效样本数量偏少，频谱分辨率有限")
    elif point_count >= 1024:
        reasons.append("有效样本数量充足，频谱识别结果具备较好的参考性")

    if sample_rate < dominant_frequency * 2.5:
        score += 1
        reasons.append("采样率相对主频偏低，建议提高采样率以降低混叠风险")

    bridge_adjustment = {
        "悬索桥": -1,
        "斜拉桥": -1,
        "拱桥": 0,
        "梁桥": 0,
        "连续梁桥": 0,
    }.get(bridge_type, 0)
    score += bridge_adjustment

    if score <= 0:
        risk_level = "安全"
        risk_color = "#22c55e"
        status = "整体振动状态平稳"
        potential_risk = "暂未发现明显潜在风险"
        advice = "建议保持常规巡检频率，持续保存原始振动数据，用于建立长期基线。"
    elif score == 1:
        risk_level = "注意"
        risk_color = "#facc15"
        status = "存在轻微波动特征"
        potential_risk = "存在一定潜在风险，建议进行复测确认"
        advice = "建议在相同测点和相似工况下进行二次采集，并与历史主频结果对比。"
    else:
        risk_level = "风险"
        risk_color = "#ef4444"
        status = "振动响应存在异常倾向"
        potential_risk = "存在潜在结构或局部构件异常风险"
        advice = "建议立即复测，并结合现场交通、风速、支座、伸缩缝和结构构件状态开展专项检查。"

    frequency_abnormal = "异常" if score >= 2 else "需关注" if score == 1 else "未见明显异常"
    feature_summary = "；".join(reasons) if reasons else "FFT 结果未显示明显异常特征"
    conclusion = (
        f"根据频谱分析结果，桥梁当前主频为 {dominant_frequency:.3f} Hz，"
        f"{frequency_status}。从幅值特征看，{feature_summary}。"
        f"综合采样质量、主频分布和桥梁类型判断，当前桥梁状态为：{status}，"
        f"风险等级评定为：{risk_level}。{potential_risk}。{advice}"
    )

    return {
        "risk_level": risk_level,
        "risk_color": risk_color,
        "status": status,
        "advice": advice,
        "summary": feature_summary,
        "frequency_note": frequency_status,
        "frequency_abnormal": frequency_abnormal,
        "potential_risk": potential_risk,
        "engineering_conclusion": conclusion,
        "score": score,
    }
