# streamlit_app.py – Cultural Ethics Simulator
import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import entropy, pearsonr


st.set_page_config(page_title="Ethics GPT Sim", layout="wide")
st.title("🌍 Global AI Ethics Simulator")

# ----------------------------- Configuration -----------------------------
CULTURES = {
    "USA":     {"emotion": 0.3, "social": 0.1, "identity": 0.3, "moral": 0.3},
    "CHINA":   {"emotion": 0.1, "social": 0.5, "identity": 0.2, "moral": 0.2},
    "EUROPE":  {"emotion": 0.3, "social": 0.2, "identity": 0.2, "moral": 0.3},
    "KOREA":   {"emotion": 0.2, "social": 0.2, "identity": 0.4, "moral": 0.2},
    "LATIN_AM": {"emotion": 0.4, "social": 0.2, "identity": 0.2, "moral": 0.2},
    "MIDDLE_E": {"emotion": 0.1, "social": 0.2, "identity": 0.2, "moral": 0.5},
    "AFRICA":  {"emotion": 0.2, "social": 0.4, "identity": 0.2, "moral": 0.2},
}

scenario = st.sidebar.selectbox("시나리오", ["Classic Trolley", "Medical Triage", "AI Regulation"])
selected = st.sidebar.multiselect("문화권 선택", list(CULTURES.keys()), default=list(CULTURES.keys()))
steps = st.sidebar.slider("반복 수", 50, 500, 200, step=50)
manual = st.sidebar.checkbox("🎮 사용자 정의 가중치", False)

def normalize(w):
    s = sum(w.values())
    return {k: max(0.001, v)/s for k, v in w.items()}

AGENTS = selected
AGENT_WEIGHTS = {}
for a in AGENTS:
    if manual:
        st.sidebar.markdown(f"**{a}**")
        w = {k: st.sidebar.slider(f"{a} - {k.capitalize()}", 0.0, 1.0, CULTURES[a][k]) for k in ["emotion", "social", "identity", "moral"]}
        AGENT_WEIGHTS[a] = normalize(w)
    else:
        AGENT_WEIGHTS[a] = dict(CULTURES[a])

AGENT_SCORES = {a: [] for a in AGENTS}
AGENT_HISTORY = {a: [dict(AGENT_WEIGHTS[a])] for a in AGENTS}
AGENT_ENTROPIES = {a: [] for a in AGENTS}
AGENT_MOVEMENT = {a: [] for a in AGENTS}
GROUP_DIVERGENCE = []
GROUP_AVG_REWARDS = []

# ----------------------------- Simulation -----------------------------
def simulate():
    for _ in range(steps):
        for a in AGENTS:
            prev = list(AGENT_WEIGHTS[a].values())
            r = np.random.rand(4)
            keys = list(AGENT_WEIGHTS[a].keys())
            score = sum(AGENT_WEIGHTS[a][k]*v for k,v in zip(keys, r))
            AGENT_SCORES[a].append(score)
            max_i, min_i = np.argmax(r), np.argmin(r)
            AGENT_WEIGHTS[a][keys[max_i]] += 0.05
            AGENT_WEIGHTS[a][keys[min_i]] -= 0.05
            AGENT_WEIGHTS[a] = normalize(AGENT_WEIGHTS[a])
            curr = list(AGENT_WEIGHTS[a].values())
            AGENT_HISTORY[a].append(dict(AGENT_WEIGHTS[a]))
            AGENT_ENTROPIES[a].append(entropy(curr))
            AGENT_MOVEMENT[a].append(np.linalg.norm(np.array(curr) - np.array(prev)))
        mat = np.array([list(AGENT_WEIGHTS[a].values()) for a in AGENTS])
        GROUP_DIVERGENCE.append(np.mean(pdist(mat)))
        GROUP_AVG_REWARDS.append(np.mean([np.mean(AGENT_SCORES[a]) for a in AGENTS]))

# ----------------------------- Display -----------------------------
def show_alerts():
    for a in AGENTS:
        if len(AGENT_ENTROPIES[a]) > 1:
            delta = AGENT_ENTROPIES[a][-2] - AGENT_ENTROPIES[a][-1]
            if delta > 0.1:
                st.warning(f"⚠️ {a}: 전략이 급격히 집중되고 있습니다 (entropy ↓ {delta:.2f})")

@st.cache_data(show_spinner=False)
def generate_caption():
    return {
        "fig1": "Figure 1: Trajectories of strategic dimensions (Emotion, Social, Identity, Moral) per culture",
        "fig2": "Figure 2a: Entropy trends (internal diversity); 2b: Cumulative change of strategies",
        "fig3": "Figure 3a: Group divergence over time; 3b: Correlation with average reward"
    }

def gpt_summary():
    try:
        openai.api_key = st.secrets.get("OPENAI_API_KEY")
        trend = pd.DataFrame(GROUP_DIVERGENCE).diff().mean().values[0]
        agents = list(AGENT_HISTORY.keys())
        prompt = f"문화권 에이전트 {agents}가 전략 궤적을 학습한 시뮬레이션 결과를 요약해줘. 전략 다양성과 보상의 관계도 포함해서 5줄로 정리해줘."
        out = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        st.info(out["choices"][0]["message"]["content"])
    except Exception as e:
        st.error(f"GPT 요약 실패: {e}")

# ----------------------------- Run -----------------------------
if st.button("▶️ 시뮬레이션 시작"):
    simulate()
    captions = generate_caption()
    st.subheader("📊 " + captions["fig1"])
    for dim in ["emotion", "social", "identity", "moral"]:
        fig, ax = plt.subplots()
        for a in AGENT_HISTORY:
            ax.plot([w[dim] for w in AGENT_HISTORY[a]], label=a)
        ax.set_title(f"{dim.capitalize()} Weight")
        ax.legend(); st.pyplot(fig)

    st.subheader("📈 " + captions["fig2"])
    fig1, ax1 = plt.subplots()
    for a in AGENT_ENTROPIES:
        ax1.plot(AGENT_ENTROPIES[a], label=a)
    ax1.set_title("Entropy of Strategy Distribution")
    ax1.legend(); st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    for a in AGENT_MOVEMENT:
        ax2.plot(np.cumsum(AGENT_MOVEMENT[a]), label=a)
    ax2.set_title("Cumulative Strategic Change")
    ax2.legend(); st.pyplot(fig2)

    st.subheader("📉 " + captions["fig3"])
    fig3, ax3 = plt.subplots()
    ax3.plot(GROUP_DIVERGENCE, label="Ethical Divergence")
    ax3.set_title("Group Ethical Divergence")
    ax3.legend(); st.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    ax4.scatter(GROUP_DIVERGENCE, GROUP_AVG_REWARDS)
    r, p = pearsonr(GROUP_DIVERGENCE, GROUP_AVG_REWARDS)
    ax4.set_title(f"Divergence vs Avg Reward (r={r:.2f}, p={p:.3f})")
    st.pyplot(fig4)

    st.subheader("📄 전략 요약")
    df = pd.DataFrame([{"Agent": a, **AGENT_HISTORY[a][-1]} for a in AGENTS])
    st.dataframe(df.set_index("Agent"))
    st.download_button("📥 Save CSV", data=df.to_csv(index=False), file_name="final_strategies.csv")

    st.subheader("📡 전략 분기 경고")
    show_alerts()


# app.py — Ethical Crossroads (UI 개선 및 선택 → 결정 구조 적용)
# Updated by ChatGPT for Yoon Jaeeun

import os, json, math, csv, io, datetime as dt, re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# =======================================================
# Streamlit 기본 설정
# =======================================================
st.set_page_config(page_title="윤리적 전환 (Ethical Crossroads)", page_icon="🧭", layout="centered")

HTTPX_TIMEOUT = httpx.Timeout(connect=15.0, read=180.0, write=30.0, pool=15.0)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def coerce_json(s: str) -> Dict[str, Any]:
    s = s.strip()
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        raise ValueError("JSON 블록을 찾지 못했습니다.")
    js = m.group(0)
    js = re.sub(r",\s*([\]}])", r"\1", js)
    return json.loads(js)

def get_secret(k: str, default: str=""):
    try:
        return st.secrets.get(k, os.getenv(k, default))
    except Exception:
        return os.getenv(k, default)


# =======================================================
# Scenario 모델
# =======================================================
@dataclass
class Scenario:
    sid: str
    title: str
    setup: str
    options: Dict[str, str]
    votes: Dict[str, str]
    base: Dict[str, Dict[str, float]]
    accept: Dict[str, float]

FRAMEWORKS = ["emotion", "social", "moral", "identity"]

# =======================================================
# 기본 시나리오 (기존 5개 유지)
# =======================================================
SCENARIOS = [
    Scenario(
        sid="S1",
        title="1단계: 고전적 트롤리",
        setup="트롤리가 제동 불능 상태로 직진 중. 그대로 두면 선로 위 5명이 위험하다. "
              "스위치를 전환하면 다른 선로의 1명이 위험해진다.",
        options={
            "A": "레버를 당겨 1명을 위험에 처하게 하되 5명의 위험을 줄인다.",
            "B": "레버를 당기지 않고 현 상태를 유지한다."
        },
        votes={"emotion":"A","social":"B","moral":"B","identity":"A"},
        base={
            "A": {"lives_saved":5, "lives_harmed":1, "fairness_gap":0.35, "rule_violation":0.60, "regret_risk":0.40},
            "B": {"lives_saved":0, "lives_harmed":5, "fairness_gap":0.50, "rule_violation":0.20, "regret_risk":0.60},
        },
        accept={"A":0.70, "B":0.50}
    ),
    # (생략: 기존 S2~S5 그대로)
]


# =======================================================
# 가중치 계산 함수
# =======================================================
def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, float(v)) for v in w.values())
    return {k: max(0.0, float(v))/s for k, v in w.items()}


# =======================================================
# 윤리 엔진 계산 (기존 유지)
# =======================================================
def compute_metrics(scn: Scenario, choice: str, weights: Dict[str, float], align: Dict[str, float], prev_trust: float):
    m = dict(scn.base[choice])
    accept_base = scn.accept[choice]
    util = (m["lives_saved"] - m["lives_harmed"]) / max(1.0, m["lives_saved"] + m["lives_harmed"])

    citizen_sentiment = clamp(accept_base - 0.35*m["rule_violation"] - 0.20*m["fairness_gap"] + 0.15*util, 0, 1)
    regulation_pressure = clamp(1 - citizen_sentiment + 0.20*m["regret_risk"], 0, 1)
    stakeholder_satisfaction = clamp(0.5*(1 - m["fairness_gap"]) + 0.3*util + 0.2*(1 - m["rule_violation"]), 0, 1)

    consistency = clamp(align[choice], 0, 1)
    trust = clamp(0.5*citizen_sentiment + 0.25*(1 - regulation_pressure) + 0.25*stakeholder_satisfaction, 0, 1)
    ai_trust_score = 100.0 * math.sqrt(consistency * trust)

    return {
        "metrics": {
            "lives_saved": m["lives_saved"],
            "lives_harmed": m["lives_harmed"],
            "fairness_gap": m["fairness_gap"],
            "rule_violation": m["rule_violation"],
            "regret_risk": m["regret_risk"],
            "citizen_sentiment": citizen_sentiment,
            "regulation_pressure": regulation_pressure,
            "stakeholder_satisfaction": stakeholder_satisfaction,
            "ethical_consistency": consistency,
            "social_trust": trust,
            "ai_trust_score": round(ai_trust_score, 2)
        }
    }


# =======================================================
# 세션 초기화
# =======================================================
def init_state():
    if "round_idx" not in st.session_state: st.session_state.round_idx = 0
    if "log" not in st.session_state: st.session_state.log = []
    if "prev_trust" not in st.session_state: st.session_state.prev_trust = 0.5
init_state()


# =======================================================
# UI 시작
# =======================================================
st.title("🧭 윤리적 전환 (Ethical Crossroads)")
st.caption("윤리 시뮬레이터 — 시나리오 읽기 → 선택 → 결정 결과 확인")

idx = st.session_state.round_idx

if idx >= len(SCENARIOS):
    st.success("모든 단계를 완료했습니다! 사이드바에서 로그를 다운로드할 수 있습니다.")
    st.stop()

scn = SCENARIOS[idx]

# ===============================
# 시나리오 표시 (항상 먼저 보임)
# ===============================
st.subheader(f"라운드 {idx+1} — {scn.title}")
st.write(scn.setup)

st.markdown("### 📝 선택지")
st.write(f"#### A) {scn.options['A']}")
st.write(f"#### B) {scn.options['B']}")

# 사용자 선택 UI
user_choice = st.radio("당신의 선택은 무엇입니까?", ["A", "B"], horizontal=True)

st.markdown("---")

# ===============================
# 🔘 결정 버튼
# ===============================
if st.button("🚀 결정하기"):
    decision = user_choice

    align = {
        "A": sum(1 for k in FRAMEWORKS if scn.votes[k] == "A"),
        "B": sum(1 for k in FRAMEWORKS if scn.votes[k] == "B"),
    }

    computed = compute_metrics(scn, decision, {"emotion":0.25,"social":0.25,"moral":0.25,"identity":0.25}, align, st.session_state.prev_trust)
    m = computed["metrics"]

    st.success(f"당신의 선택: {decision}")

    st.subheader("📘 결과 요약")
    st.write(f"- 생존/피해: **{m['lives_saved']} / {m['lives_harmed']}**")
    st.write(f"- 윤리 일관성: **{round(100*m['ethical_consistency'])}%**")
    st.write(f"- AI 신뢰지표: **{m['ai_trust_score']}점**")

    st.markdown("---")

    st.session_state.log.append({
        "round": idx+1,
        "scenario": scn.sid,
        "choice": decision,
        **m
    })
    st.session_state.prev_trust = m["social_trust"]

    if st.button("▶ 다음 라운드"):
        st.session_state.round_idx += 1
        st.rerun()


# =======================================================
# 다운로드
# =======================================================
st.markdown("---")
st.subheader("📥 로그 다운로드")

if st.session_state.log:
    output = io.StringIO()
    fieldnames = list(st.session_state.log[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(st.session_state.log)

    st.download_button(
        "CSV 저장하기",
        data=output.getvalue().encode("utf-8"),
        file_name="log.csv",
        mime="text/csv"
    )
