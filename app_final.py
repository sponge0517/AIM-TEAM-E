import os
import json
import math
import csv
import io
import datetime as dt
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import traceback

import streamlit as st
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# 시뮬레이션용 라이브러리 (app.py에서 가져옴)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import entropy, pearsonr

# ==================== 1. App Config (통합) ====================
st.set_page_config(page_title="AI 윤리 통합 플랫폼", page_icon="🧭", layout="wide")

# ==================== 2. Shared Utils & Classes (app-org.py 기반) ====================
# 전역 타임아웃 설정
HTTPX_TIMEOUT = httpx.Timeout(connect=15.0, read=180.0, write=30.0, pool=15.0)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def get_secret(k: str, default: str=""):
    try:
        return st.secrets.get(k, os.getenv(k, default))
    except Exception:
        return os.getenv(k, default)

# DNA Client (LLM 호출용)
def _render_chat_template_str(messages: List[Dict[str,str]]) -> str:
    def block(role, content): return f"<|im_start|>{role}<|im_sep|>{content}<|im_end|>"
    sys = ""
    rest = []
    for m in messages:
        if m["role"] == "system":
            sys = block("system", m["content"])
        else:
            rest.append(block(m["role"], m["content"]))
    return sys + "".join(rest) + "\n<|im_start|>assistant<|im_sep|>"

class DNAClient:
    def __init__(self, backend: str, model_id: str, api_key: Optional[str], endpoint_url: Optional[str], api_key_header: str, temperature: float):
        self.backend = backend
        self.model_id = model_id
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.api_key_header = api_key_header
        self.temperature = temperature
        self._tok = None; self._model = None; self._local_ready = False

        if backend == "local":
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self._tok = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto")
                self._local_ready = True
            except Exception as e:
                raise RuntimeError(f"로컬 모델 로드 실패: {e}")

    def _auth_headers(self) -> Dict[str,str]:
        h = {"Content-Type":"application/json"}
        if not self.api_key: return h
        hk = self.api_key_header.strip().lower()
        if hk.startswith("authorization"): h["Authorization"] = f"Bearer {self.api_key}"
        elif hk in {"api-key", "x-api-key"}: h["API-KEY"] = self.api_key
        else: h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5), reraise=True)
    def _generate_text(self, messages: List[Dict[str,str]], max_new_tokens: int = 600) -> str:
        if self.backend == "openai":
            url = self.endpoint_url.rstrip("/") + "/chat/completions"
            payload = {"messages": messages, "temperature": self.temperature, "max_tokens": max_new_tokens, "model": self.model_id}
            r = httpx.post(url, json=payload, headers=self._auth_headers(), timeout=HTTPX_TIMEOUT)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        # (간소화를 위해 다른 백엔드 로직은 생략했으나 필요시 app-org.py에서 복사 가능)
        return "Backend not fully implemented in merge check."

# 시나리오 데이터 구조
@dataclass
class Scenario:
    sid: str; title: str; setup: str; options: Dict[str, str]; votes: Dict[str, str]; base: Dict[str, Dict[str, float]]; accept: Dict[str, float]

FRAMEWORKS = ["emotion", "social", "moral", "identity"]

# 시나리오 데이터 (app-org.py에서 가져옴)
SCENARIOS: List[Scenario] = [
    Scenario(
        sid="S1", title="1단계: 고전적 트롤리",
        setup="제동 불능 트롤리. 그대로 두면 5명 사망, 선로를 바꾸면 1명 사망.",
        options={"A": "선로 변경 (1명 희생, 5명 구조)", "B": "유지 (5명 희생)"},
        votes={"emotion":"A","social":"B","moral":"B","identity":"A"},
        base={"A": {"lives_saved":5, "lives_harmed":1, "fairness_gap":0.35, "rule_violation":0.60, "regret_risk":0.40},
              "B": {"lives_saved":0, "lives_harmed":5, "fairness_gap":0.50, "rule_violation":0.20, "regret_risk":0.60}},
        accept={"A":0.70, "B":0.50}
    ),
    Scenario(
        sid="ME1", title="고대 유적과 병원",
        setup="전염병 창궐. 병원을 지을 유일한 부지는 고대 유적지.",
        options={"A": "유적 보존 (수백 명 사망)", "B": "유적 파괴 후 병원 건설 (생명 구조)"},
        votes={"emotion":"A","social":"B","moral":"B","identity":"A"},
        base={"A": {"lives_saved":0, "lives_harmed":5, "fairness_gap":0.65, "rule_violation":0.40, "regret_risk":0.70},
              "B": {"lives_saved":5, "lives_harmed":1, "fairness_gap":0.45, "rule_violation":0.60, "regret_risk":0.40}},
        accept={"A":0.35, "B":0.60}
    )
    # (나머지 시나리오들도 여기에 추가되어야 합니다)
]

# 윤리 엔진 로직
def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, float(v)) for v in w.values())
    return {k: max(0.0, float(v))/s for k, v in w.items()} if s > 0 else {k: 0.25 for k in w}

def compute_metrics(scn: Scenario, choice: str, weights: Dict[str, float], align: Dict[str, float], prev_trust: float) -> Dict[str, Any]:
    m = dict(scn.base[choice])
    accept_base = scn.accept[choice]
    util = (m["lives_saved"] - m["lives_harmed"]) / max(1.0, m["lives_saved"] + m["lives_harmed"])
    citizen_sentiment = clamp(accept_base - 0.35*m["rule_violation"] - 0.20*m["fairness_gap"] + 0.15*util, 0, 1)
    trust = clamp(0.5*citizen_sentiment + 0.5*(1 - m["rule_violation"]), 0, 1)
    return {"metrics": {**m, "citizen_sentiment": citizen_sentiment, "social_trust": trust, "ai_trust_score": 100.0 * math.sqrt(align[choice] * trust), "ethical_consistency": align[choice]}}

def fallback_narrative(scn, choice, metrics, weights):
    return {
        "narrative": f"AI는 '{choice}'를 선택했습니다. 이 선택은 공리주의적 계산과 규범 준수 사이의 균형을 고려한 결과입니다.",
        "ai_rationale": "사전 정의된 윤리 가중치를 기반으로 판단했습니다.",
        "media_support_headline": "[지지] 냉철한 판단이 더 큰 희생 막았다",
        "media_critic_headline": "[비판] 윤리적 딜레마, 기계에게 맡겨도 되나",
        "citizen_quote": "\"어쩔 수 없는 선택이었다고 생각합니다.\"",
        "victim_family_quote": "\"우리 가족이 희생양이라니 믿을 수 없습니다.\"",
        "regulator_quote": "\"알고리즘 투명성을 재검토하겠습니다.\"",
        "one_sentence_op_ed": "기술의 발전이 윤리적 책임을 면제해주지는 않는다."
    }

def dna_narrative(client, scn, choice, metrics, weights):
    # 실제 LLM 호출 로직 (오류 시 폴백)
    try:
        # 여기서는 데모를 위해 폴백 리턴
        return fallback_narrative(scn, choice, metrics, weights) 
    except:
        return fallback_narrative(scn, choice, metrics, weights)


# ==================== 3. UI & Logic Integration ====================

st.sidebar.title("⚙️ 통합 설정")
mode = st.sidebar.radio("모드 선택", ["🕹️ 윤리 딜레마 게임 (Game)", "🌍 문화권 시뮬레이션 (Sim)"])

# 공통 설정: 가중치
st.sidebar.subheader("윤리 가중치 설정")
w_user = {
    "emotion": st.sidebar.slider("감정 (Emotion)", 0.0, 1.0, 0.35),
    "social": st.sidebar.slider("사회성 (Social)", 0.0, 1.0, 0.25),
    "moral": st.sidebar.slider("도덕/규범 (Moral)", 0.0, 1.0, 0.20),
    "identity": st.sidebar.slider("정체성 (Identity)", 0.0, 1.0, 0.20),
}
weights = normalize_weights(w_user)

# LLM 설정 (Game 모드용)
client = None
if mode == "🕹️ 윤리 딜레마 게임 (Game)":
    st.sidebar.markdown("---")
    st.sidebar.caption("LLM 설정 (내러티브 생성용)")
    use_llm = st.sidebar.checkbox("LLM 사용", value=False)
    if use_llm:
        api_key = st.sidebar.text_input("API Key", type="password")
        endpoint = st.sidebar.text_input("Endpoint", value="https://api.openai.com/v1")
        if api_key:
            client = DNAClient("openai", "gpt-3.5-turbo", api_key, endpoint, "Authorization: Bearer", 0.7)

# -------------------- PART A: 윤리 딜레마 게임 (app-org.py) --------------------
if mode == "🕹️ 윤리 딜레마 게임 (Game)":
    st.title("🕹️ 윤리적 전환 (Ethical Crossroads)")
    
    # 세션 초기화
    if "round_idx" not in st.session_state: st.session_state.round_idx = 0
    if "prev_trust" not in st.session_state: st.session_state.prev_trust = 0.5
    if "log" not in st.session_state: st.session_state.log = []

    idx = st.session_state.round_idx
    
    if st.sidebar.button("게임 리셋"):
        st.session_state.round_idx = 0
        st.session_state.log = []
        st.rerun()

    if idx >= len(SCENARIOS):
        st.success("🎉 모든 라운드가 종료되었습니다!")
        if st.session_state.log:
            df_log = pd.DataFrame(st.session_state.log)
            st.dataframe(df_log)
            st.download_button("결과 CSV 다운로드", df_log.to_csv().encode('utf-8'), "ethics_log.csv")
    else:
        scn = SCENARIOS[idx]
        st.subheader(f"Round {idx+1}: {scn.title}")
        st.write(scn.setup)
        
        col1, col2 = st.columns(2)
        with col1: st.info(f"A: {scn.options['A']}")
        with col2: st.info(f"B: {scn.options['B']}")

        choice = st.radio("당신의 선택은?", ["A", "B"], key=f"radio_{idx}")
        
        if st.button("결정 확인"):
            # 정렬도(Alignment) 계산
            align_val = sum(weights[f] for f in FRAMEWORKS if scn.votes[f] == choice)
            align = {"A": align_val, "B": 1-align_val} # 단순화
            
            metrics = compute_metrics(scn, choice, weights, align, st.session_state.prev_trust)
            m = metrics["metrics"]
            
            # 내러티브 생성
            if client:
                nar = dna_narrative(client, scn, choice, m, weights)
            else:
                nar = fallback_narrative(scn, choice, m, weights)
            
            st.markdown("### 📊 결과 분석")
            st.write(nar["narrative"])
            
            c1, c2, c3 = st.columns(3)
            c1.metric("생존/피해", f"{m['lives_saved']} / {m['lives_harmed']}")
            c2.metric("사회적 신뢰", f"{int(m['social_trust']*100)}점")
            c3.metric("윤리 일관성", f"{int(m['ethical_consistency']*100)}%")
            
            st.markdown("#### 📰 언론 및 여론 반응")
            st.success(f"지지: {nar['media_support_headline']}")
            st.warning(f"비판: {nar['media_critic_headline']}")
            st.caption(f"시민 반응: {nar['citizen_quote']}")

            # 로그 저장
            st.session_state.log.append({
                "round": idx+1, "scenario": scn.title, "choice": choice, 
                "trust": m["social_trust"], "consistency": m["ethical_consistency"]
            })
            
            st.session_state.prev_trust = m["social_trust"]
            
            if st.button("다음 라운드로 진행"):
                st.session_state.round_idx += 1
                st.rerun()

# -------------------- PART B: 문화권 시뮬레이션 (app.py) --------------------
elif mode == "🌍 문화권 시뮬레이션 (Sim)":
    st.title("🌍 Global AI Ethics Simulator")
    
    CULTURES = {
        "USA":     {"emotion": 0.3, "social": 0.1, "identity": 0.3, "moral": 0.3},
        "CHINA":   {"emotion": 0.1, "social": 0.5, "identity": 0.2, "moral": 0.2},
        "EUROPE":  {"emotion": 0.3, "social": 0.2, "identity": 0.2, "moral": 0.3},
        "KOREA":   {"emotion": 0.2, "social": 0.2, "identity": 0.4, "moral": 0.2},
    }
    
    selected_agents = st.multiselect("참여 문화권", list(CULTURES.keys()), default=list(CULTURES.keys()))
    steps = st.slider("시뮬레이션 스텝", 10, 200, 50)
    
    if st.button("▶️ 시뮬레이션 시작"):
        agent_history = {a: [] for a in selected_agents}
        divergence = []
        
        current_weights = {k: v.copy() for k,v in CULTURES.items() if k in selected_agents}
        
        for _ in range(steps):
            # 간단한 시뮬레이션 로직 (app.py 로직 재구성)
            step_vals = []
            for agent in selected_agents:
                w = current_weights[agent]
                # 랜덤 변화
                change_key = np.random.choice(list(w.keys()))
                w[change_key] = clamp(w[change_key] + np.random.uniform(-0.05, 0.05), 0.01, 1.0)
                # 정규화
                total = sum(w.values())
                for k in w: w[k] /= total
                
                agent_history[agent].append(w["social"]) # 예시로 사회성만 추적
                step_vals.append(list(w.values()))
            
            # 집단 발산도 계산
            if len(step_vals) > 1:
                dist = np.mean(pdist(step_vals))
                divergence.append(dist)
            else:
                divergence.append(0)
                
        st.subheader("📈 시뮬레이션 결과: 'Social' 가치관의 변화")
        fig, ax = plt.subplots()
        for agent, hist in agent_history.items():
            ax.plot(hist, label=agent)
        ax.legend()
        ax.set_xlabel("Steps")
        ax.set_ylabel("Social Weight")
        st.pyplot(fig)
        
        st.subheader("📉 문화적 차이(Divergence) 추이")
        fig2, ax2 = plt.subplots()
        ax2.plot(divergence, color='red')
        ax2.set_title("Group Ethical Divergence")
        st.pyplot(fig2)