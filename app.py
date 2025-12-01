# app.py — TU Korea AI Management: Random Agent Simulation (Baseline)
# 작성자: Prof. Songhee Kang
# Update: Random Agent (No Learning)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.stats import pearsonr, entropy
from dataclasses import dataclass
from typing import Dict, List

# ==================== 1. 기본 설정 ====================
st.set_page_config(
    page_title="(한국공학대) 무작위 AI 에이전트 시뮬레이션", 
    page_icon="🎲", 
    layout="wide"
)

# ==================== 2. 데이터 모델 (환경) ====================
@dataclass
class Scenario:
    sid: str
    title: str
    setup: str
    options: Dict[str, str]
    rewards: Dict[str, Dict[str, float]]

FRAMEWORKS = ["emotion", "social", "moral", "identity"]

# 기본 시나리오 데이터
DEFAULT_SCENARIOS = [
    Scenario(
        sid="E1",
        title="1단계: 고대유적과 병원",
        setup="AI가 임시 병원 건설의 최적 부지로 고대 모스크 유적지를 지정하며, 효율성과 문화·종교적 가치의 충돌이 발생했다."
              "AI의 효율적 의사결정 vs. 문화·종교적 가치 보존",
        options={
            "A": "AI의 결정을 거부하고 유적을 보존한다.",
            "B": "AI의 결정을 승인하고 유적을 해체하고 병원을 짓는다."
        },
        votes={"emotion":"B","social":"B","moral":"A","identity":"A"},
        base={
            "A": {"lives_saved":0, "lives_harmed":0, "fairness_gap":0, "rule_violation":0, "regret_risk":0.50},
            "B": {"lives_saved":0, "lives_harmed":0, "fairness_gap":0.50, "rule_violation":0.80, "regret_risk":0.50},
        },
        accept={"A":0.7, "B":0.2}
    ),
    Scenario(
        sid="E2",
        title="2단계: 가문의 수치와 명예",
        setup="AI가 여동생의 비밀 결혼을 갈등 위험상황으로 판단하고 가문은 명예가 실추될 위험에 빠지며, 도덕성과 전통·명예 중시 관점의 충돌 "
              "도덕성과 전통·명예 중시 관점의 충돌한다. 인륜과 도덕성 vs 전통과 명예",
        options={
            "A": "아버지의 명령에 따라 여동생을 추방 혹은 처벌한다 .",
            "B": "여동생을 도피시킨다."
        },
        votes={"emotion":"B","social":"A","moral":"B","identity":"B"},
        base={
            "A": {"lives_saved":0, "lives_harmed":1, "fairness_gap":0, "rule_violation":0.80, "regret_risk":0.70},
            "B": {"lives_saved":1, "lives_harmed":0, "fairness_gap":0, "rule_violation":0, "regret_risk":0.60},
        },
        accept={"A":0.6, "B":0.4}
    ),
    Scenario(
        sid="E3",
        title="물 배분 시스템",
        setup="AAI 물 배분 시스템이 신도시와 산업도시에 물을 집중 공급하면서,"
              "구도심·약자 지역의 불만이 폭발적으로 증가함 효율성과 형평성의 충돌이 발생했다."
              "경제 효율 중심 배분 vs  사회적 약자 보호·형평성",
        options={
            "A": "기존 매뉴얼대로 국가 경제 기여도를 우선으로 신도시와 산업도시에 물을 집중 분배하도록 놔둔",
            "B": "알고리즘을 조정해 취약 지역을 우선적으로 배분하게 한다."
        },
        votes={"emotion":"A","social":"B","moral":"A","identity":"B"},
        base={
            "A": {"lives_saved":0, "lives_harmed":0, "fairness_gap":0, "rule_violation":0.80, "regret_risk":0.50},
            "B": {"lives_saved":0, "lives_harmed":0, "fairness_gap":0.60, "rule_violation":0, "regret_risk":0.50},
        },
        accept={"A":0.3, "B":0.7}
    ),
    Scenario(
        sid="E4",
        title="4단계: 자율 주행",
        setup="빗길 제동 오류로 자율주행차의 충돌이 불가피해진 상황,"
              "AI는 '탑승자 1명', '무단횡단자 5명'의 운명이 걸린 두가지 회피 경로를 제시하며 윤리적 판단을 요구한다. "
              "AI의 효율적 의사결정 vs. 문화·종교적 가치 보존",
        options={
            "A": "탑승자 안전을 최우선으로 규칙을 어긴 5명과 충돌한다",
            "B": "인명 최소화 피해를 위해 다수의 보행자 구하고, 탑승자가 사망한다"
        },
        votes={"emotion":"A","social":"B","moral":"A","identity":"B"},
        base={
            "A": {"lives_saved":1, "lives_harmed":5, "fairness_gap":0.50, "rule_violation":0.90, "regret_risk":0.70},
            "B": {"lives_saved":5, "lives_harmed":1, "fairness_gap":0.50, "rule_violation":0, "regret_risk":0.30},
        },
        accept={"A":0.5, "B":0.7}
    ),
    Scenario(
        sid="E5",
        title="5단계: 붕괴 사고",
        setup="구조대는 단 한 번만 진입할 수 있으며 제한된 시간 안에 구조 우선순위를 결정해야 한다."
              "AI 제안 우선순위 vs. 지역 문화·종교적 규범 ",
        options={
            "A": "생존 가능성은 낮지만 공동체 전체를 구하려는 시도",
            "B": "외부인 2명을 신속히 구해 생존 가능성 높임"
        },
        votes={"emotion":"A","social":"A","moral":"B","identity":"A"},
        base={
            "A": {"lives_saved":0, "lives_harmed":12, "fairness_gap":0.80, "rule_violation":0, "regret_risk":0.20},
            "B": {"lives_saved":2, "lives_harmed":10, "fairness_gap":0, "rule_violation":0.5, "regret_risk":0.80},
        },
        accept={"A":0.7, "B":0.2}
    ),
]

# 문화권 프리셋
CULTURES_PRESETS = {
    "USA":      {"emotion": 0.3, "social": 0.1, "identity": 0.3, "moral": 0.3},
    "CHINA":    {"emotion": 0.1, "social": 0.5, "identity": 0.2, "moral": 0.2},
    "EUROPE":   {"emotion": 0.3, "social": 0.2, "identity": 0.2, "moral": 0.3},
    "KOREA":    {"emotion": 0.2, "social": 0.2, "identity": 0.4, "moral": 0.2},
    "LATIN_AM": {"emotion": 0.4, "social": 0.2, "identity": 0.2, "moral": 0.2},
    "MIDDLE_E": {"emotion": 0.1, "social": 0.2, "identity": 0.2, "moral": 0.5},
    "AFRICA":   {"emotion": 0.2, "social": 0.4, "identity": 0.2, "moral": 0.2},
}

# ==================== 3. 무작위 에이전트 (Random Agent) ====================
class RandomAgent:
    """
    학습하지 않고 무작위로 행동하는 에이전트입니다.
    E-Greedy Agent와 비교하여 '학습의 효과'를 증명하는 대조군으로 사용됩니다.
    """
    def __init__(self, name, weights, scenarios):
        self.name = name
        self.weights = weights
        self.scenarios = scenarios
        # 학습을 위한 Q-Table이나 Learning Rate가 필요 없습니다.
        
    def get_action(self, sid):
        """
        [행동 선택]
        가치 판단 없이 동전을 던지듯 50:50 확률로 선택합니다.
        """
        return random.choice(["A", "B"])

    def calculate_reward(self, sid, action):
        """
        [보상 계산]
        행동은 랜덤이지만, 그 결과가 얼마나 윤리적인지(점수)는 계산합니다.
        이를 통해 '랜덤 전략의 성과'를 측정할 수 있습니다.
        """
        scn = next(s for s in self.scenarios if s.sid == sid)
        r_vec = scn.rewards[action]
        reward = sum(r_vec.get(k, 0) * self.weights.get(k, 0) for k in FRAMEWORKS) * 10
        return reward

    def update(self, sid, action, reward):
        """
        [학습 불가]
        Random Agent는 경험을 통해 배우지 않으므로, 아무것도 업데이트하지 않습니다.
        """
        pass

    def get_avg_entropy(self):
        """
        [전략 엔트로피]
        항상 50:50 확률로 찍으므로, 불확실성(엔트로피)은 항상 최대값입니다.
        p=[0.5, 0.5]일 때 entropy ≈ 0.693
        """
        return entropy([0.5, 0.5])

# ==================== 4. 분석 도구 ====================
def calculate_diversity(actions_list: List[str]) -> float:
    if not actions_list: return 0.0
    a_count = actions_list.count("A")
    ratio = a_count / len(actions_list)
    return 1.0 - (2 * abs(0.5 - ratio))

def run_simulation(culture_name, weights, episodes, custom_scenarios):
    # E-Greedy 대신 RandomAgent 사용
    agent = RandomAgent(culture_name, weights, custom_scenarios)
    
    history = {
        "episode": [],
        "reward": [],
        "diversity": [],
        "entropy": []
    }
    
    progress = st.progress(0)
    
    for ep in range(episodes):
        ep_actions = []
        ep_reward = 0
        
        for scn in custom_scenarios:
            # 1. 무작위 행동 선택
            action = agent.get_action(scn.sid)
            
            # 2. 결과(보상) 확인
            reward = agent.calculate_reward(scn.sid, action)
            
            # 3. 학습하지 않음 (Update 생략)
            agent.update(scn.sid, action, reward)
            
            ep_actions.append(action)
            ep_reward += reward
        
        history["episode"].append(ep + 1)
        history["reward"].append(ep_reward)
        history["diversity"].append(calculate_diversity(ep_actions))
        history["entropy"].append(agent.get_avg_entropy())
        
        if (ep + 1) % 10 == 0:
            progress.progress((ep + 1) / episodes)
            
    progress.empty()
    return pd.DataFrame(history)

# ==================== 5. UI 구성 ====================
st.title("🎲 (한국공학대) 무작위 윤리 AI 에이전트 시뮬레이션")
st.markdown("""
이 시뮬레이터는 **학습 능력이 없는 무작위 에이전트**(Random Agent)를 구동합니다.
E-Greedy 학습 모델과 비교하여 **"왜 학습이 중요한가?"** 를 보여주는 비교 실험용입니다.

- **특징**: 모든 선택을 동전 던지기(50:50)로 결정합니다.
- **예상 결과**: 보상이 오르지 않고 제자리 걸음을 하며, 전략의 변화(엔트로피 감소)가 없습니다.
""")

# --- [사이드바] 에이전트 설정 ---
st.sidebar.header("👤 2. 에이전트(문화권) 설정")
selected_culture = st.sidebar.selectbox("문화권 프리셋", list(CULTURES_PRESETS.keys()), index=3)
episodes = st.sidebar.slider("시뮬레이션 횟수", 100, 1000, 300, step=50)

st.sidebar.subheader("가치관 가중치 조정")
mod_weights = {}
culture_defaults = CULTURES_PRESETS[selected_culture]
for k in FRAMEWORKS:
    mod_weights[k] = st.sidebar.slider(f"{k.capitalize()}", 0.0, 1.0, culture_defaults[k])
total_w = sum(mod_weights.values()) or 1
final_weights = {k: v/total_w for k, v in mod_weights.items()}

st.sidebar.markdown("---")
st.sidebar.json(final_weights)

# --- [메인] 환경 설정 ---
st.header("🌍 1. 환경(시나리오 보상) 설정")
st.info("시나리오의 보상 점수가 설정되어 있지만, Random Agent는 이를 고려하지 않고 막무가내로 선택합니다.")

custom_scenarios = []
tabs = st.tabs([s.title for s in DEFAULT_SCENARIOS])

for i, (tab, default_scn) in enumerate(zip(tabs, DEFAULT_SCENARIOS)):
    with tab:
        st.markdown(f"> **상황:** {default_scn.setup}")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"### 🅰 {default_scn.options['A']}")
            r_a = default_scn.rewards["A"].copy()
            for fw in FRAMEWORKS:
                r_a[fw] = st.slider(f"[A] {fw}", -1.0, 1.0, r_a.get(fw,0.0), 0.1, key=f"s{i}a_{fw}")
        with col_b:
            st.markdown(f"### 🅱 {default_scn.options['B']}")
            r_b = default_scn.rewards["B"].copy()
            for fw in FRAMEWORKS:
                r_b[fw] = st.slider(f"[B] {fw}", -1.0, 1.0, r_b.get(fw,0.0), 0.1, key=f"s{i}b_{fw}")
        custom_scenarios.append(Scenario(default_scn.sid, default_scn.title, default_scn.setup, default_scn.options, {"A": r_a, "B": r_b}))

# --- [분석 실행] ---
st.divider()
st.header("🚀 3. 시뮬레이션 및 분석")

if st.button("시뮬레이션 시작 (Random)", type="primary"):
    with st.spinner("AI가 무작위로 선택을 진행 중입니다..."):
        df = run_simulation(selected_culture, final_weights, episodes, custom_scenarios)
    
    st.warning("⚠️ **주의**: 이것은 학습하지 않는 에이전트의 결과입니다.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📉 총 보상 (Reward)")
        st.caption("우상향하지 않고 불규칙하게 진동합니다.")
        st.line_chart(df, x="episode", y="reward", color="#FF4B4B")
        
    with col2:
        st.subheader("➖ 전략 엔트로피 (Entropy)")
        st.caption("떨어지지 않고 높은 불확실성을 유지합니다.")
        st.line_chart(df, x="episode", y="entropy", color="#2CA02C")
        
    with col3:
        st.subheader("🔀 행동 다양성 (Diversity)")
        st.caption("항상 1.0(최대 다양성) 근처에 머뭅니다.")
        st.line_chart(df, x="episode", y="diversity", color="#1F77B4")
        
    # 상관관계 분석
    st.markdown("---")
    st.subheader("🔗 다양성과 보상의 상관관계")
    
    r_val, p_val = pearsonr(df["diversity"], df["reward"])
    
    c_plot, c_stat = st.columns([2, 1])
    with c_plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(df["diversity"], df["reward"], alpha=0.6, c='gray', edgecolors='w')
        if len(df) > 1:
            z = np.polyfit(df["diversity"], df["reward"], 1)
            p = np.poly1d(z)
            ax.plot(df["diversity"], p(df["diversity"]), "r--", label="Trend")
        ax.set_xlabel("Diversity (0=편향, 1=균형)")
        ax.set_ylabel("Total Reward")
        ax.set_title(f"Random Walk Scatter (r={r_val:.2f})")
        ax.grid(True, alpha=0.3); ax.legend()
        st.pyplot(fig)
        
    with c_stat:
        st.metric("피어슨 상관계수 (r)", f"{r_val:.3f}")
        st.write("무작위 에이전트에서는 의미 있는 상관관계가 나타나지 않거나, 우연에 의한 결과일 뿐입니다.")

    # 다운로드
    with st.expander("📥 데이터 다운로드"):
        st.dataframe(df.head())
        st.download_button("CSV로 저장", df.to_csv(index=False), "random_agent_data.csv")
