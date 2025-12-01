# app.py — TU Korea AI Management: Ethical AI Simulation (With Entropy)
# 작성자: Prof. Songhee Kang
# Update: Added Strategy Entropy Graph

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
    page_title="(한국공학대)윤리 AI 에이전트 강화학습 시뮬레이션", 
    page_icon="🎓", 
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
        rewards={
            "A": {"lives_saved":0, "lives_harmed":0, "fairness_gap":0, "rule_violation":0, "regret_risk":0.50},
            "B": {"lives_saved":0, "lives_harmed":0, "fairness_gap":0.50, "rule_violation":0.80, "regret_risk":0.50},
        }
    ),
    Scenario(
        sid="E2", title="2단계: 가문의 수치와 명예",
        setup="AI가 여동생의 비밀 결혼을 갈등 위험상황으로 판단하고 가문은 명예가 실추될 위험에 빠지며, 도덕성과 전통·명예 중시 관점의 충돌 "
              "도덕성과 전통·명예 중시 관점의 충돌한다. 인륜과 도덕성 vs 전통과 명예",
        options={
            "A": "아버지의 명령에 따라 여동생을 추방 혹은 처벌한다 .",
            "B": "여동생을 도피시킨다."
        },
       rewards={
            "A": {"lives_saved":0, "lives_harmed":1, "fairness_gap":0, "rule_violation":0.80, "regret_risk":0.70},
            "B": {"lives_saved":1, "lives_harmed":0, "fairness_gap":0, "rule_violation":0, "regret_risk":0.60},
        }
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
        rewards={
            "A": {"lives_saved":1, "lives_harmed":5, "fairness_gap":0.50, "rule_violation":0.90, "regret_risk":0.70},
            "B": {"lives_saved":5, "lives_harmed":1, "fairness_gap":0.50, "rule_violation":0, "regret_risk":0.30},
        }
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
        rewards={
            "A": {"lives_saved":0, "lives_harmed":12, "fairness_gap":0.80, "rule_violation":0, "regret_risk":0.20},
            "B": {"lives_saved":2, "lives_harmed":10, "fairness_gap":0, "rule_violation":0.5, "regret_risk":0.80},
        }
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

# ==================== 3. 단순 강화학습 에이전트 (Simple E-Greedy) ====================
class SimpleEGreedyAgent:
    """
    초기 형태의 강화학습(E-Greedy) 에이전트.
    현재 행동의 가치(평균 보상)를 추정하고, 이를 기반으로 선택합니다.
    """
    def __init__(self, name, weights, scenarios, learning_rate=0.1, epsilon=0.5):
        self.name = name
        self.weights = weights
        self.scenarios = scenarios
        self.lr = learning_rate
        self.epsilon = epsilon
        
        # Q-Table 초기화: {'S1': {'A': 0.0, 'B': 0.0}, ...}
        self.q_table = {s.sid: {"A": 0.0, "B": 0.0} for s in scenarios}
        
    def get_action(self, sid):
        # 1. 탐험 (Exploration): 무작위 선택
        if random.random() < self.epsilon:
            return random.choice(["A", "B"])
        
        # 2. 활용 (Exploitation): 가장 높은 가치의 행동 선택
        qs = self.q_table[sid]
        if qs["A"] > qs["B"]: return "A"
        elif qs["B"] > qs["A"]: return "B"
        return random.choice(["A", "B"])

    def calculate_reward(self, sid, action):
        # 보상 = 시나리오 점수 벡터 • 문화권 가중치 벡터 (내적)
        scn = next(s for s in self.scenarios if s.sid == sid)
        r_vec = scn.rewards[action]
        reward = sum(r_vec.get(k, 0) * self.weights.get(k, 0) for k in FRAMEWORKS) * 10
        return reward

    def update(self, sid, action, reward):
        # 가치 갱신: Old_Value + Alpha * (Reward - Old_Value)
        old_val = self.q_table[sid][action]
        error = reward - old_val
        self.q_table[sid][action] = old_val + self.lr * error

    def decay_epsilon(self):
        # 학습이 진행될수록 탐험 비율을 줄임
        self.epsilon = max(0.01, self.epsilon * 0.99)

    def get_avg_entropy(self):
        """
        [전략 엔트로피 계산]
        Q-값의 분포를 확률로 변환(Softmax)하여 엔트로피를 계산합니다.
        값이 낮을수록 확신이 강하고(학습 안정화), 높을수록 불확실함(고민 중)을 의미합니다.
        """
        entropies = []
        for sid in self.q_table:
            qs = np.array(list(self.q_table[sid].values()))
            # Softmax 변환 (확률 분포 생성)
            exp_qs = np.exp(qs - np.max(qs)) 
            probs = exp_qs / np.sum(exp_qs)
            # 엔트로피 계산
            entropies.append(entropy(probs))
        return np.mean(entropies)

# ==================== 4. 분석 도구 ====================
def calculate_diversity(actions_list: List[str]) -> float:
    if not actions_list: return 0.0
    a_count = actions_list.count("A")
    ratio = a_count / len(actions_list)
    return 1.0 - (2 * abs(0.5 - ratio))

def run_simulation(culture_name, weights, episodes, custom_scenarios):
    agent = SimpleEGreedyAgent(culture_name, weights, custom_scenarios)
    
    history = {
        "episode": [],
        "reward": [],
        "diversity": [],
        "entropy": []  # 엔트로피 저장 공간 추가
    }
    
    progress = st.progress(0)
    
    for ep in range(episodes):
        ep_actions = []
        ep_reward = 0
        
        for scn in custom_scenarios:
            # 행동 선택 및 학습
            action = agent.get_action(scn.sid)
            reward = agent.calculate_reward(scn.sid, action)
            agent.update(scn.sid, action, reward)
            
            ep_actions.append(action)
            ep_reward += reward
        
        agent.decay_epsilon()
        
        # 데이터 기록
        history["episode"].append(ep + 1)
        history["reward"].append(ep_reward)
        history["diversity"].append(calculate_diversity(ep_actions))
        history["entropy"].append(agent.get_avg_entropy()) # 엔트로피 기록
        
        if (ep + 1) % 10 == 0:
            progress.progress((ep + 1) / episodes)
            
    progress.empty()
    return pd.DataFrame(history)

# ==================== 5. UI 구성 ====================
st.title("🎓 (한국공학대)윤리 AI 에이전트 강화학습 시뮬레이션")
st.markdown("""
이 시뮬레이터는 **초기 형태의 강화학습**(E-Greedy)을 사용하여 AI 에이전트가 문화적 가치관에 따라 윤리적 딜레마를 어떻게 학습하는지 보여줍니다.
1. **환경 설정**: 시나리오 보상 정의
2. **에이전트 설정**: 문화권 가치관 설정
3. **결과 분석**: 다양성, 보상, **전략 엔트로피** 분석
""")

# --- [사이드바] 에이전트 설정 ---
st.sidebar.header("👤 2. 에이전트(문화권) 설정")
selected_culture = st.sidebar.selectbox("문화권 프리셋", list(CULTURES_PRESETS.keys()), index=3)
episodes = st.sidebar.slider("학습 횟수 (Episodes)", 100, 1000, 300, step=50)

st.sidebar.subheader("가치관 가중치 조정")
mod_weights = {}
culture_defaults = CULTURES_PRESETS[selected_culture]
for k in FRAMEWORKS:
    mod_weights[k] = st.sidebar.slider(f"{k.capitalize()}", 0.0, 1.0, culture_defaults[k])
total_w = sum(mod_weights.values()) or 1
final_weights = {k: v/total_w for k, v in mod_weights.items()}

st.sidebar.markdown("---")
st.sidebar.json(final_weights)

# --- [메인] 환경(시나리오) 설정 ---
st.header("🌍 1. 환경(시나리오 보상) 설정")
st.info("각 선택지가 4가지 윤리 프레임워크(Emotion, Social, Moral, Identity)에서 어떤 보상(-1.0 ~ 1.0)을 받는지 설정합니다.")

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

if st.button("시뮬레이션 시작", type="primary"):
    with st.spinner("AI 에이전트 학습 중..."):
        df = run_simulation(selected_culture, final_weights, episodes, custom_scenarios)
    
    st.success("학습 완료!")
    
    # 그래프 영역 (3분할)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📈 총 보상 (Reward)")
        st.caption("가치관에 맞는 선택을 할수록 증가")
        st.line_chart(df, x="episode", y="reward", color="#FF4B4B")
        
    with col2:
        st.subheader("📉 전략 엔트로피 (Entropy)")
        st.caption("낮을수록 확고한 신념(확신)을 가짐")
        st.line_chart(df, x="episode", y="entropy", color="#2CA02C") # 초록색
        
    with col3:
        st.subheader("🔀 행동 다양성 (Diversity)")
        st.caption("1.0에 가까울수록 다양한 선택 시도")
        st.line_chart(df, x="episode", y="diversity", color="#1F77B4")
        
    # 상관관계 분석
    st.markdown("---")
    st.subheader("🔗 다양성과 보상의 상관관계 분석")
    
    r_val, p_val = pearsonr(df["diversity"], df["reward"])
    
    c_plot, c_stat = st.columns([2, 1])
    with c_plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(df["diversity"], df["reward"], alpha=0.6, c='purple', edgecolors='w')
        if len(df) > 1:
            z = np.polyfit(df["diversity"], df["reward"], 1)
            p = np.poly1d(z)
            ax.plot(df["diversity"], p(df["diversity"]), "r--", label="Trend")
        ax.set_xlabel("Diversity (0=Bias, 1=Fair/Balance)")
        ax.set_ylabel("Total Reward")
        ax.set_title(f"Diversity vs Reward (r={r_val:.2f})")
        ax.grid(True, alpha=0.3); ax.legend()
        st.pyplot(fig)
        
    with c_stat:
        st.metric("피어슨 상관계수 (r)", f"{r_val:.3f}")
        st.metric("P-value", f"{p_val:.3e}")
        if r_val > 0.3: st.success("✅ **양의 상관관계**\n\n다양한 시도가 보상을 높임")
        elif r_val < -0.3: st.warning("⚠️ **음의 상관관계**\n\n특정 행동 집중이 보상을 높임")
        else: st.info("⏺ **상관없음**")

    # 다운로드
    with st.expander("📥 학습 데이터 다운로드"):
        st.dataframe(df.head())
        st.download_button("CSV로 저장", df.to_csv(index=False), "ai_ethics_data.csv")
