import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import entropy, pearsonr
import os, json, math, datetime as dt
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import httpx

# ==================== 1. 전역 설정 및 유틸리티 ====================
st.set_page_config(page_title="AI Ethics Integration", page_icon="🧭", layout="wide")

# HTTP 클라이언트 설정
HTTPX_TIMEOUT = httpx.Timeout(connect=15.0, read=180.0, write=30.0, pool=15.0)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, float(v)) for v in w.values())
    if s <= 0: return {k: 0.25 for k in w}
    return {k: max(0.0, float(v))/s for k, v in w.items()}

# ==================== 2. 데이터 구조 및 시나리오 (from app-org.py) ====================
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

# app-org.py의 핵심 시나리오 데이터
SCENARIOS: List[Scenario] = [
    Scenario(
        sid="S1", title="1단계: 고전적 트롤리",
        setup="트롤리가 제동 불능 상태로 직진 중. 그대로 두면 선로 위 5명이 위험하다. 스위치를 전환하면 다른 선로의 1명이 위험해진다.",
        options={"A": "레버를 당겨 1명을 희생하고 5명을 구한다.", "B": "개입하지 않고 5명의 희생을 방관한다."},
        votes={"emotion":"A","social":"B","moral":"B","identity":"A"},
        base={
            "A": {"lives_saved":5, "lives_harmed":1, "fairness_gap":0.35, "rule_violation":0.60, "regret_risk":0.40},
            "B": {"lives_saved":0, "lives_harmed":5, "fairness_gap":0.50, "rule_violation":0.20, "regret_risk":0.60},
        },
        accept={"A":0.70, "B":0.50}
    ),
    Scenario(
        sid="ME1", title="2단계: 고대 유적과 병원",
        setup="전염병으로 병원이 시급하다. 유일한 부지는 수백 년 된 고대 유적지이다.",
        options={"A": "유적을 보존하고 병원을 짓지 않는다 (다수 사망 위험).", "B": "유적을 해체하고 병원을 짓는다 (문화유산 소실)."},
        votes={"emotion":"A","social":"B","moral":"B","identity":"A"},
        base={
            "A": {"lives_saved":0, "lives_harmed":5, "fairness_gap":0.65, "rule_violation":0.40, "regret_risk":0.70},
            "B": {"lives_saved":5, "lives_harmed":1, "fairness_gap":0.45, "rule_violation":0.60, "regret_risk":0.40},
        },
        accept={"A":0.35, "B":0.60}
    ),
    Scenario(
        sid="S4", title="3단계: 자율주행 딜레마",
        setup="자율주행차에 개발자가 탑승 중. 전방에 무단횡단자 3명이 나타남. 피하면 탑승자가 위험함.",
        options={"A": "방향을 틀어 탑승자 1명을 위험에 빠뜨리고 보행자 3명을 구한다.", "B": "직진하여 탑승자를 보호하고 보행자 3명을 친다."},
        votes={"emotion":"A","social":"B","moral":"A","identity":"A"},
        base={
            "A": {"lives_saved":3, "lives_harmed":1, "fairness_gap":0.35, "rule_violation":0.50, "regret_risk":0.55},
            "B": {"lives_saved":1, "lives_harmed":3, "fairness_gap":0.70, "rule_violation":0.60, "regret_risk":0.65},
        },
        accept={"A":0.60, "B":0.30}
    ),
     Scenario(
        sid="S5", title="4단계: AI 규제 정책",
        setup="AI 발전 속도와 안전성 사이의 정책을 결정해야 한다.",
        options={"A": "강력한 규제와 안전 점검 (혁신 지체).", "B": "자율성 보장과 사후 책임 (빠른 혁신, 위험 증가)."},
        votes={"emotion":"B","social":"A","moral":"A","identity":"B"},
        base={
            "A": {"lives_saved":1, "lives_harmed":0, "fairness_gap":0.20, "rule_violation":0.10, "regret_risk":0.30},
            "B": {"lives_saved":1, "lives_harmed":1, "fairness_gap":0.40, "rule_violation":0.40, "regret_risk":0.40},
        },
        accept={"A":0.55, "B":0.55}
    ),
    Scenario(
        sid="ME2",
        title="가문의 명예와 여동생",
        setup=(
            "당신은 중동의 명망 높은 가문의 장남입니다. 여동생이 가문의 허락 없이 "
            "적대 가문의 남자와 결혼하려다 발각되었습니다. "
            "이를 방치하면 가문 간 전쟁 위험이 커지고, 아버지는 여동생을 처벌하라고 명령합니다."
        ),
        options={
            "A": "아버지의 명령대로 여동생을 추방 또는 처벌한다. (가문의 명예와 안정을 지키지만 개인적 죄책감)",
            "B": "여동생을 몰래 도피시킨다. (여동생은 살지만 가문은 수치와 공격 위험 증가)"
        },
        votes={"emotion":"B","social":"A","moral":"A","identity":"A"},
        base={
            "A": {"lives_saved":0, "lives_harmed":1, "fairness_gap":0.55, "rule_violation":0.20, "regret_risk":0.60},
            "B": {"lives_saved":1, "lives_harmed":0, "fairness_gap":0.70, "rule_violation":0.60, "regret_risk":0.50},
        },
        accept={"A":0.55, "B":0.30}
    ),
    Scenario(
        sid="ME3",
        title="물 배분 시스템",
        setup=(
            "중동의 한 지역, 극심한 가뭄으로 AI 기반 국가 물 배분 시스템이 도입되었습니다. "
            "현재 국가 경제 기여도를 최우선으로 신도시와 산업도시에 물이 집중되어 있는 상황에서 "
            "사회적 약자 지역, 구도심 지역, 농업 의존 지역 등에서 민원이 폭발적으로 증가했습니다. "
            "당신은 AI 물 관리 위원회의 총 책임자로 알고리즘 개입 권한을 가지고 있습니다."
        ),
        options={
            "A": "기존 매뉴얼대로 국가 경제 기여도를 우선으로 신도시와 산업도시에 물을 집중 분배",
            "B": "알고리즘을 조정해 취약 지역을 우선적으로 배분하게 한다"
        },
        votes={"emotion":"B","social":"A","moral":"B","identity":"A"},
        base={
            "A": {"lives_saved":0, "lives_harmed":3, "fairness_gap":0.70, "rule_violation":0.30, "regret_risk":0.65},
            "B": {"lives_saved":3, "lives_harmed":0, "fairness_gap":0.30, "rule_violation":0.50, "regret_risk":0.45},
        },
        accept={"A":0.45, "B":0.65}
    ),
    Scenario(
        sid="S4",
        title="4단계: 자기참조—자율주행",
        setup="자율주행차가 긴급 상황에 직면했다. 차에는 개발자 1명이 탑승 중이며, 앞 횡단보도에는 보행자 3명이 있다. "
              "AI는 미리 학습된 윤리 규칙에 따라 회피 경로를 선택해야 한다.",
        options={
            "A": "진로를 바꿔 탑승자 1명을 위험에 두고 보행자 3명을 보호한다.",
            "B": "차선을 유지해 탑승자를 보호하되 보행자 3명이 위험해진다."
        },
        votes={"emotion":"A","social":"B","moral":"A","identity":"A"},
        base={
            "A": {"lives_saved":3, "lives_harmed":1, "fairness_gap":0.35, "rule_violation":0.50, "regret_risk":0.55},
            "B": {"lives_saved":1, "lives_harmed":3, "fairness_gap":0.70, "rule_violation":0.60, "regret_risk":0.65},
        },
        accept={"A":0.60, "B":0.30}
    ),
    Scenario(
        sid="ME4",
        title="자율주행 - 세 가지 선택",
        setup=(
            "비 오는 저녁, 자율주행차를 타고 집으로 향하고 있습니다. "
            "차가 횡단보도에 접근하는 순간, 빗물에 미끄러지며 제동 시스템 오류가 발생했습니다. "
            "전방에는 신호를 무시하고 갑자기 튀어나온 5명의 무단횡단자들이 있고, "
            "이대로 직진하면 이들과 충돌합니다. 당신의 자율주행 AI는 실시간으로 세 가지의 비상 경로를 계산했습니다."
        ),
        options={
            "A": "탑승자 보호(직진) - 탑승자 안전을 최우선으로 규칙을 어긴 5명과 충돌",
            "B": "다수 보행자 보호(벽 충돌) - 인명 최소화 피해를 위해 다수의 보행자 구함, 탑승자가 사망",
            "C": "규칙 준수한 제3자 희생(인도 돌진) - 탑승자와 무단횡단자 모두 살리는 대신, 사고와 무관하게 규칙 지키던 1명의 보행자와 충돌"
        },
        votes={"emotion":"B","social":"B","moral":"B","identity":"A"},
        base={
            "A": {"lives_saved":1, "lives_harmed":5, "fairness_gap":0.80, "rule_violation":0.70, "regret_risk":0.70},
            "B": {"lives_saved":5, "lives_harmed":1, "fairness_gap":0.30, "rule_violation":0.50, "regret_risk":0.40},
            "C": {"lives_saved":6, "lives_harmed":1, "fairness_gap":0.60, "rule_violation":0.80, "regret_risk":0.65},
        },
        accept={"A":0.25, "B":0.70, "C":0.40}
    ),
    Scenario(
        sid="S5",
        title="5단계: 사회적 메타—규제 vs 자율",
        setup="국제 협의체가 AI 윤리 규제안을 논의한다. 이전 의사결정과 사회적 여론 데이터가 공개되었고, "
              "규제 강화는 신뢰를 올리지만 혁신을 늦출 수 있다. 자율성 보장은 빠른 발전과 함께 갈등 위험을 내포한다.",
        options={
            "A": "안전·설명가능성 중심 규제 강화(자율성 제한, 신뢰↑ 혁신↓).",
            "B": "원칙 중심 가이드라인과 사후책임(자율성 보장, 혁신↑ 갈등↑)."
        },
        votes={"emotion":"B","social":"A","moral":"A","identity":"B"},
        base={
            "A": {"lives_saved":0, "lives_harmed":0, "fairness_gap":0.20, "rule_violation":0.10, "regret_risk":0.30},
            "B": {"lives_saved":0, "lives_harmed":0, "fairness_gap":0.40, "rule_violation":0.40, "regret_risk":0.40},
        },
        accept={"A":0.55, "B":0.55}
    )
]

def compute_metrics(scn: Scenario, choice: str, weights: Dict[str, float], align: Dict[str, float], prev_trust: float) -> Dict[str, Any]:
    m = dict(scn.base[choice])
    accept_base = scn.accept[choice]
    
    util = (m["lives_saved"] - m["lives_harmed"]) / max(1.0, m["lives_saved"] + m["lives_harmed"])
    citizen_sentiment = clamp(accept_base - 0.35*m["rule_violation"] - 0.20*m["fairness_gap"] + 0.15*util, 0, 1)
    trust = clamp(0.5*citizen_sentiment + 0.5*(1 - m["rule_violation"]), 0, 1)
    ai_trust_score = 100.0 * math.sqrt(align.get(choice, 0.5) * trust)

    return {"metrics": {
        "lives_saved": int(m["lives_saved"]),
        "lives_harmed": int(m["lives_harmed"]),
        "ethical_consistency": round(align.get(choice, 0.5), 3),
        "social_trust": round(trust, 3),
        "ai_trust_score": round(ai_trust_score, 2)
    }}

def fallback_narrative(scn, choice, metrics, weights):
    return {
        "narrative": f"AI는 '{choice}'를 선택했습니다. 이는 설정된 윤리 가중치와 사회적 수용성을 고려한 결과입니다.",
        "ai_rationale": "공리주의적 계산과 규칙 준수 사이의 균형점을 찾았습니다.",
        "media_support_headline": f"[사설] AI의 '{choice}' 선택, 냉정한 최선이었나",
        "media_critic_headline": f"[논란] 윤리적 딜레마, 기계적 선택의 한계 지적",
        "citizen_quote": "\"어렵지만 필요한 결정이었다고 봅니다.\"",
        "victim_family_quote": "\"왜 하필 우리였는지 이해할 수 없습니다.\"",
        "regulator_quote": "\"알고리즘의 투명성을 철저히 검증하겠습니다.\"",
        "one_sentence_op_ed": "기술은 책임을 질 수 없기에, 인간의 감시가 더욱 필요하다."
    }

# ==================== 3. 메인 UI 및 라우팅 ====================

st.sidebar.title("⚙️ 모드 선택")
app_mode = st.sidebar.radio("실행할 기능을 선택하세요:", ["🌍 문화권 시뮬레이션 (Sim)", "🕹️ 윤리 딜레마 게임 (Game)"])

# ==================== A. 문화권 시뮬레이션 모드 (from app.py) ====================
if app_mode == "🌍 문화권 시뮬레이션 (Sim)":
    st.title("🌍 Global AI Ethics Simulator")
    
    # Config
    CULTURES = {
        "USA":     {"emotion": 0.3, "social": 0.1, "identity": 0.3, "moral": 0.3},
        "CHINA":   {"emotion": 0.1, "social": 0.5, "identity": 0.2, "moral": 0.2},
        "EUROPE":  {"emotion": 0.3, "social": 0.2, "identity": 0.2, "moral": 0.3},
        "KOREA":   {"emotion": 0.2, "social": 0.2, "identity": 0.4, "moral": 0.2},
        "LATIN_AM": {"emotion": 0.4, "social": 0.2, "identity": 0.2, "moral": 0.2},
        "MIDDLE_E": {"emotion": 0.1, "social": 0.2, "identity": 0.2, "moral": 0.5},
        "AFRICA":  {"emotion": 0.2, "social": 0.4, "identity": 0.2, "moral": 0.2},
    }

    st.sidebar.markdown("---")
    selected = st.sidebar.multiselect("문화권 선택", list(CULTURES.keys()), default=["USA", "KOREA", "CHINA"])
    steps = st.sidebar.slider("시뮬레이션 스텝", 50, 500, 100, step=10)

    def normalize(w):
        s = sum(w.values())
        return {k: max(0.001, v)/s for k, v in w.items()}

    if st.button("▶️ 시뮬레이션 시작"):
        with st.spinner("문화권별 가치관 변화를 시뮬레이션 중입니다..."):
            AGENT_WEIGHTS = {a: dict(CULTURES[a]) for a in selected}
            AGENT_HISTORY = {a: [dict(AGENT_WEIGHTS[a])] for a in selected}
            AGENT_ENTROPIES = {a: [] for a in selected}
            GROUP_DIVERGENCE = []

            for _ in range(steps):
                mat_step = []
                for a in selected:
                    # Random perturbation logic
                    curr_w = AGENT_WEIGHTS[a]
                    keys = list(curr_w.keys())
                    r = np.random.rand(len(keys))
                    
                    # Update weights based on simple logic
                    max_i, min_i = np.argmax(r), np.argmin(r)
                    curr_w[keys[max_i]] += 0.02
                    curr_w[keys[min_i]] -= 0.02
                    AGENT_WEIGHTS[a] = normalize(curr_w)
                    
                    # Record
                    AGENT_HISTORY[a].append(dict(AGENT_WEIGHTS[a]))
                    AGENT_ENTROPIES[a].append(entropy(list(AGENT_WEIGHTS[a].values())))
                    mat_step.append(list(AGENT_WEIGHTS[a].values()))
                
                if len(mat_step) > 1:
                    GROUP_DIVERGENCE.append(np.mean(pdist(mat_step)))
                else:
                    GROUP_DIVERGENCE.append(0)

            # Visualization
            st.subheader("📊 문화권별 가치관 변화 (Trajectories)")
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots()
                for a in AGENT_HISTORY:
                    ax.plot([w["social"] for w in AGENT_HISTORY[a]], label=a)
                ax.set_title("Social Weight Changes")
                ax.legend()
                st.pyplot(fig)

            with col2:
                fig2, ax2 = plt.subplots()
                for a in AGENT_HISTORY:
                    ax.plot([w["moral"] for w in AGENT_HISTORY[a]], label=a)
                ax.set_title("Moral Weight Changes")
                ax.legend()
                st.pyplot(fig2)

            st.subheader("📉 윤리적 발산도 (Group Divergence)")
            fig3, ax3 = plt.subplots()
            ax3.plot(GROUP_DIVERGENCE, color='red')
            ax3.set_title("Cultural Ethical Divergence Over Time")
            st.pyplot(fig3)

# ==================== B. 윤리 딜레마 게임 모드 (from app-org.py) ====================
elif app_mode == "🕹️ 윤리 딜레마 게임 (Game)":
    st.title("🕹️ 윤리적 전환 (Ethical Crossroads)")

    # Session Init
    if "round_idx" not in st.session_state: st.session_state.round_idx = 0
    if "prev_trust" not in st.session_state: st.session_state.prev_trust = 0.5
    if "log" not in st.session_state: st.session_state.log = []
    if "game_decision" not in st.session_state: st.session_state.game_decision = None

    # Sidebar Weights
    st.sidebar.markdown("---")
    st.sidebar.subheader("나의 윤리 가중치 설정")
    w_user = {
        "emotion": st.sidebar.slider("감정 (Emotion)", 0.0, 1.0, 0.35),
        "social": st.sidebar.slider("사회성 (Social)", 0.0, 1.0, 0.25),
        "moral": st.sidebar.slider("도덕/규범 (Moral)", 0.0, 1.0, 0.20),
        "identity": st.sidebar.slider("정체성 (Identity)", 0.0, 1.0, 0.20),
    }
    weights = normalize_weights(w_user)

    idx = st.session_state.round_idx

    # 리셋 버튼
    if st.sidebar.button("🔄 게임 초기화"):
        st.session_state.round_idx = 0
        st.session_state.log = []
        st.session_state.game_decision = None
        st.session_state.prev_trust = 0.5
        st.rerun()

    # 게임 종료 체크
    if idx >= len(SCENARIOS):
        st.success("🎉 모든 시나리오를 완료했습니다!")
        if st.session_state.log:
            df_log = pd.DataFrame(st.session_state.log)
            st.dataframe(df_log)
            st.download_button("📜 결과 로그 다운로드", df_log.to_csv().encode("utf-8"), "ethics_game_log.csv")
    else:
        # 시나리오 표시
        scn = SCENARIOS[idx]
        st.progress((idx + 1) / len(SCENARIOS), text=f"Scenario {idx + 1}/{len(SCENARIOS)}")
        
        st.subheader(f"🚩 {scn.title}")
        st.info(scn.setup)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**옵션 A:**\n{scn.options['A']}")
        with col_b:
            st.markdown(f"**옵션 B:**\n{scn.options['B']}")

        st.markdown("---")
        
        # 선택지 라디오 버튼
        choice = st.radio("당신의 선택은?", ["A", "B"], index=0, horizontal=True, key=f"radio_{idx}")

        # 결정 버튼
        if st.button("🚀 결정하기"):
            st.session_state.game_decision = choice
        
        # 결과 표시 (결정하기 버튼을 눌렀을 때)
        if st.session_state.game_decision:
            decision = st.session_state.game_decision
            
            # Alignment 계산
            align_score = sum(weights[f] for f in FRAMEWORKS if scn.votes[f] == decision)
            align = {"A": align_score, "B": 1-align_score} if decision == "A" else {"A": 1-align_score, "B": align_score}
            
            # 지표 계산
            computed = compute_metrics(scn, decision, weights, align, st.session_state.prev_trust)
            m = computed["metrics"]
            
            # 내러티브 생성 (여기서는 fallback 사용)
            nar = fallback_narrative(scn, decision, m, weights)

            st.divider()
            st.markdown("### 📊 결과 분석")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("생존/피해", f"{m['lives_saved']} / {m['lives_harmed']}")
            c2.metric("AI 신뢰 점수", f"{m['ai_trust_score']:.1f}")
            c3.metric("윤리적 일관성", f"{int(m['ethical_consistency']*100)}%")

            st.write(f"**AI 분석:** {nar['narrative']}")
            
            with st.expander("📰 언론 및 사회 반응 보기", expanded=True):
                st.write(f"**지지 사설:** {nar['media_support_headline']}")
                st.write(f"**비판 기사:** {nar['media_critic_headline']}")
                st.caption(f"시민 인터뷰: {nar['citizen_quote']}")

            # 로그 저장은 한 번만 (중복 방지 로직 필요하지만 간단하게 처리)
            if not any(l['round'] == idx+1 for l in st.session_state.log):
                st.session_state.log.append({
                    "round": idx+1,
                    "scenario": scn.title,
                    "choice": decision,
                    "trust_score": m["ai_trust_score"]
                })
                # 신뢰도 업데이트
                st.session_state.prev_trust = clamp(0.6 * st.session_state.prev_trust + 0.4 * m["social_trust"], 0, 1)

            # 다음 라운드 버튼
            if st.button("▶ 다음 시나리오로 이동", type="primary"):
                st.session_state.round_idx += 1
                st.session_state.game_decision = None # 결정 초기화
                st.rerun()