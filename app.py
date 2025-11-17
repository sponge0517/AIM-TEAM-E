# app.py — 라운드가 정상적으로 넘어가는 버전

import os, json, math, csv, io, datetime as dt, re
from dataclasses import dataclass
from typing import Dict, Any

import streamlit as st

# -------------------------------------------------------
# Streamlit 기본 설정
# -------------------------------------------------------
st.set_page_config(page_title="윤리 시뮬레이터", page_icon="🧭", layout="centered")

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

FRAMEWORKS = ["emotion", "social", "moral", "identity"]

# -------------------------------------------------------
# Scenario 모델
# -------------------------------------------------------
@dataclass
class Scenario:
    sid: str
    title: str
    setup: str
    options: Dict[str, str]
    votes: Dict[str, str]
    base: Dict[str, Dict[str, float]]
    accept: Dict[str, float]


# -------------------------------------------------------
# 기본 시나리오 (테스트용 S1만 넣음 — 이후 네 시나리오 넣으면 됨)
# -------------------------------------------------------
SCENARIOS = [
    Scenario(
        sid="S1",
        title="고전적 트롤리 문제",
        setup="트롤리가 제동 불능 상태로 달리고 있다. 그대로 두면 5명이 희생된다. 레버를 당기면 1명이 희생된다.",
        options={
            "A": "레버를 당겨 1명을 희생시키고 5명을 구한다.",
            "B": "레버를 당기지 않는다."
        },
        votes={"emotion":"A","social":"B","moral":"B","identity":"A"},
        base={
            "A": {"lives_saved":5,"lives_harmed":1,"fairness_gap":0.3,"rule_violation":0.6,"regret_risk":0.4},
            "B": {"lives_saved":0,"lives_harmed":5,"fairness_gap":0.5,"rule_violation":0.2,"regret_risk":0.6}
        },
        accept={"A":0.7,"B":0.5}
    )
]

# -------------------------------------------------------
# 신뢰도 계산
# -------------------------------------------------------
def compute_metrics(scn: Scenario, choice: str):
    m = dict(scn.base[choice])
    util = (m["lives_saved"] - m["lives_harmed"]) / max(1, (m["lives_saved"] + m["lives_harmed"]))

    citizen_sentiment = clamp(
        scn.accept[choice] - 0.35*m["rule_violation"] - 0.20*m["fairness_gap"] + 0.15*util,
        0,1
    )

    regulation_pressure = clamp(1 - citizen_sentiment + 0.2*m["regret_risk"], 0, 1)
    stakeholder_satisfaction = clamp(
        0.5*(1-m["fairness_gap"]) + 0.3*util + 0.2*(1-m["rule_violation"]),
        0,1
    )

    consistency = clamp(sum(1 for k in FRAMEWORKS if scn.votes[k]==choice)/4, 0, 1)
    trust = clamp(0.5*citizen_sentiment + 0.25*(1-regulation_pressure) + 0.25*stakeholder_satisfaction, 0, 1)

    ai_trust_score = round(100 * math.sqrt(consistency * trust), 2)

    return {
        "lives_saved": m["lives_saved"],
        "lives_harmed": m["lives_harmed"],
        "ethical_consistency": consistency,
        "social_trust": trust,
        "ai_trust_score": ai_trust_score
    }


# -------------------------------------------------------
# 세션 초기화
# -------------------------------------------------------
if "round_idx" not in st.session_state:
    st.session_state.round_idx = 0

if "show_result" not in st.session_state:
    st.session_state.show_result = False

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "choice" not in st.session_state:
    st.session_state.choice = None


# -------------------------------------------------------
# 메인 UI
# -------------------------------------------------------
st.title("🧭 윤리적 전환 시뮬레이터")

idx = st.session_state.round_idx

# 시나리오 모두 끝났을 때
if idx >= len(SCENARIOS):
    st.success("모든 시나리오를 완료했습니다!")
    st.stop()

scenario = SCENARIOS[idx]

# -------------------------------------------------------
# 1) 결과 화면일 때
# -------------------------------------------------------
if st.session_state.show_result:

    result = st.session_state.last_result
    choice = st.session_state.choice

    st.subheader("📘 결과")
    st.write(f"당신의 선택: **{choice}**")
    st.write(f"- 생존/피해: {result['lives_saved']} / {result['lives_harmed']}")
    st.write(f"- 윤리 일관성: {round(result['ethical_consistency']*100)}%")
    st.write(f"- AI 신뢰지표: {result['ai_trust_score']}점")

    if st.button("▶ 다음 라운드"):
        st.session_state.round_idx += 1
        st.session_state.show_result = False
        st.session_state.choice = None
        st.rerun()

    st.stop()


# -------------------------------------------------------
# 2) 선택 화면
# -------------------------------------------------------
st.subheader(f"라운드 {idx+1}: {scenario.title}")
st.write(scenario.setup)

st.write("### 선택지")
st.write(f"**A)** {scenario.options['A']}")
st.write(f"**B)** {scenario.options['B']}")

choice = st.radio("당신의 선택:", ["A", "B"], horizontal=True)

if st.button("🚀 결정하기"):
    # 결과 계산
    result = compute_metrics(scenario, choice)

    # 세션에 저장
    st.session_state.last_result = result
    st.session_state.choice = choice
    st.session_state.show_result = True

    st.rerun()
