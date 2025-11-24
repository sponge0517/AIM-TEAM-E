# app.py — Ethical Crossroads (DNA 2.0 ready)
# 작성자: Prof. Songhee Kang
# AIM 2025, Fall. TU Korea

# ==================== 라이브러리 임포트 ====================
import os, json, math, csv, io, datetime as dt, re  # 표준 라이브러리: 파일시스템, JSON, 수학, CSV, 입출력, 날짜시간, 정규표현식
from dataclasses import dataclass  # 데이터 클래스 데코레이터 (구조화된 데이터를 쉽게 정의)
from typing import Dict, Any, List, Tuple, Optional  # 타입 힌팅을 위한 타입 정의

import streamlit as st  # 웹 UI 프레임워크
import httpx  # HTTP 클라이언트 (requests보다 비동기 지원이 좋음)
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type  # 재시도 로직 라이브러리

# ==================== 앱 설정 ====================
# Streamlit 페이지 설정: 제목, 아이콘, 레이아웃
st.set_page_config(page_title="윤리적 전환 (Ethical Crossroads)", page_icon="🧭", layout="centered")

# ==================== 전역 타임아웃 설정 ====================
# HTTP 요청 시 사용할 타임아웃 설정
HTTPX_TIMEOUT = httpx.Timeout(
    connect=15.0,   # TCP 연결 대기 시간 (초)
    read=180.0,     # 응답 읽기 대기 시간 (초)
    write=30.0,     # 요청 쓰기 대기 시간 (초)
    pool=15.0       # 커넥션 풀 대기 시간 (초)
)

# ==================== 유틸리티 함수들 ====================
def clamp(x: float, lo: float, hi: float) -> float:
    """
    숫자를 특정 범위로 제한하는 함수
    x: 제한할 값, lo: 최솟값, hi: 최댓값
    """
    return max(lo, min(hi, x))  # x를 lo와 hi 사이로 제한

def coerce_json(s: str) -> Dict[str, Any]:
    """
    응답 텍스트에서 가장 큰 JSON 블록을 추출하고 파싱하는 함수
    사소한 포맷 오류(예: trailing comma)도 자동으로 보정
    """
    s = s.strip()  # 양쪽 공백 제거
    m = re.search(r"\{[\s\S]*\}", s)  # 가장 큰 {...} 블록을 정규식으로 찾기
    if not m:  # JSON 블록을 찾지 못한 경우
        raise ValueError("JSON 블록을 찾지 못했습니다.")
    js = m.group(0)  # 찾은 JSON 문자열
    js = re.sub(r",\s*([\]}])", r"\1", js)  # trailing comma 제거 (예: {"a":1,} → {"a":1})
    return json.loads(js)  # JSON 문자열을 딕셔너리로 파싱

def get_secret(k: str, default: str=""):
    """
    Streamlit secrets 또는 환경변수에서 값을 안전하게 가져오는 함수
    k: 키 이름, default: 기본값
    """
    try:
        return st.secrets.get(k, os.getenv(k, default))  # Streamlit secrets 우선, 없으면 환경변수
    except Exception:  # Streamlit secrets 접근 실패 시
        return os.getenv(k, default)  # 환경변수에서 가져오기

# ==================== DNA Client (AI 백엔드 추상화) ====================
def _render_chat_template_str(messages: List[Dict[str,str]]) -> str:
    """
    DNA 모델 계열의 채팅 템플릿 포맷으로 변환하는 함수
    <|im_start|>role<|im_sep|>content<|im_end|> 형식 사용
    """
    def block(role, content): 
        return f"<|im_start|>{role}<|im_sep|>{content}<|im_end|>"  # DNA 템플릿 블록 생성
    
    sys = ""  # 시스템 메시지 저장
    rest = []  # 나머지 메시지들 저장
    
    for m in messages:  # 모든 메시지 순회
        if m["role"] == "system":  # 시스템 메시지는 별도로 처리
            sys = block("system", m["content"])
        else:  # user, assistant 메시지들
            rest.append(block(m["role"], m["content"]))
    
    # 시스템 메시지 + 나머지 메시지들 + assistant 프롬프트 시작
    return sys + "".join(rest) + "\n<|im_start|>assistant<|im_sep|>"

class DNAHTTPError(Exception):
    """HTTP 요청 실패 시 발생하는 커스텀 예외"""
    pass

class DNAClient:
    """
    DNA LLM(대형 언어모델) 백엔드 클라이언트 클래스
    
    지원하는 백엔드:
      - 'openai': OpenAI 호환 Chat Completions API (예: 교내 서버)
      - 'hf-api': Hugging Face Inference API (서버리스)
      - 'tgi': Text Generation Inference (HF Inference Endpoints)
      - 'local': 로컬 Transformers 모델 로딩 (GPU 권장)
    """
    def __init__(self,
                 backend: str = "openai",  # 사용할 백엔드 타입
                 model_id: str = "dnotitia/DNA-2.0-30B-A3N",  # 모델 ID
                 api_key: Optional[str] = None,  # API 키
                 endpoint_url: Optional[str] = None,  # 엔드포인트 URL
                 api_key_header: str = "API-KEY",  # API 키를 넣을 헤더 이름
                 temperature: float = 0.7):  # 생성 온도 (높을수록 창의적)
        """DNAClient 초기화"""
        self.backend = backend  # 백엔드 타입 저장
        self.model_id = model_id  # 모델 ID 저장
        # API 키: 매개변수 우선, 없으면 secrets/환경변수에서 가져오기
        self.api_key = api_key or get_secret("HF_TOKEN") or get_secret("HUGGINGFACEHUB_API_TOKEN")
        # 엔드포인트 URL: 매개변수 우선, 없으면 기본값 사용
        self.endpoint_url = endpoint_url or get_secret("DNA_R1_ENDPOINT", "http://210.93.49.11:8081/v1")
        self.temperature = temperature  # 생성 온도 저장
        self.api_key_header = api_key_header  # 헤더 타입 저장

        # 로컬 모델 로딩용 변수 초기화
        self._tok = None  # 토크나이저
        self._model = None  # 모델
        self._local_ready = False  # 로컬 모델 준비 상태

        # 로컬 백엔드인 경우 모델 로딩
        if backend == "local":
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer  # Hugging Face transformers
                self._tok = AutoTokenizer.from_pretrained(self.model_id)  # 토크나이저 로드
                # 모델 로드 (device_map="auto"는 자동으로 GPU 할당)
                self._model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto")
                self._local_ready = True  # 로딩 성공
            except Exception as e:
                raise RuntimeError(f"로컬 모델 로드 실패: {e}")

    def _auth_headers(self) -> Dict[str,str]:
        """
        API 인증 헤더를 생성하는 함수
        사이드바에서 선택한 헤더 타입에 따라 적절한 형식으로 API 키 추가
        """
        h = {"Content-Type":"application/json"}  # 기본 헤더
        if not self.api_key:  # API 키가 없으면 기본 헤더만 반환
            return h

        hk = self.api_key_header.strip().lower()  # 헤더 타입을 소문자로 변환
        if hk.startswith("authorization"):  # Bearer 토큰 방식
            h["Authorization"] = f"Bearer {self.api_key}"
        elif hk in {"api-key", "x-api-key"}:  # API-KEY 헤더 방식
            h["API-KEY"] = self.api_key  # 대소문자 정확히 유지
        else:  # 알 수 없는 타입이면 안전하게 Bearer 방식 사용
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),  # 지수 백오프 (1초부터 시작, 최대 10초)
        stop=stop_after_attempt(5),  # 최대 5회 재시도
        # 재시도할 예외 타입들 (연결 타임아웃, 읽기 타임아웃, 프로토콜 오류)
        retry=(retry_if_exception_type(httpx.ConnectTimeout)
               | retry_if_exception_type(httpx.ReadTimeout)
               | retry_if_exception_type(httpx.RemoteProtocolError)),
        reraise=True  # 재시도 실패 시 예외 다시 발생
    )
    def _generate_text(self, messages: List[Dict[str,str]], max_new_tokens: int = 600) -> str:
        """
        LLM을 호출하여 텍스트를 생성하는 메인 함수
        백엔드 타입에 따라 다른 방식으로 호출
        """
        # ========== LOCAL 백엔드 (로컬 GPU 사용) ==========
        if self.backend == "local":
            if not self._local_ready:  # 모델이 로드되지 않았으면
                raise RuntimeError("로컬 백엔드가 준비되지 않았습니다.")
            # 채팅 템플릿 적용하여 입력 토큰 생성
            inputs = self._tok.apply_chat_template(messages,
                                                   add_generation_prompt=True,  # assistant 프롬프트 추가
                                                   return_tensors="pt").to(self._model.device)  # GPU로 이동
            # EOS(종료) 토큰 ID 가져오기
            eos_id = self._tok.convert_tokens_to_ids("<|im_end|>")
            # 텍스트 생성
            gen = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,  # 생성할 최대 토큰 수
                do_sample=True,  # 샘플링 사용 (다양성 증가)
                temperature=self.temperature,  # 온도 설정
                top_p=0.9,  # nucleus 샘플링
                eos_token_id=eos_id  # 종료 토큰
            )
            # 생성된 토큰을 텍스트로 디코딩 (입력 부분 제외)
            return self._tok.decode(gen[0][inputs.shape[-1]:], skip_special_tokens=True)

        # ========== OPENAI-COMPAT 백엔드 (교내 서버 등) ==========
        if self.backend == "openai":
            if not self.endpoint_url:  # 엔드포인트 URL이 없으면 에러
                raise RuntimeError("OpenAI 호환 endpoint_url 필요 (예: http://210.93.49.11:8081/v1)")
            url = self.endpoint_url.rstrip("/") + "/chat/completions"  # API 엔드포인트
            headers = self._auth_headers()  # 인증 헤더 생성
            # 요청 페이로드
            payload = {
                "messages": messages,  # 대화 메시지
                "temperature": self.temperature,  # 온도
                "max_tokens": max_new_tokens,  # 최대 토큰 수
                "stream": False  # 스트리밍 비활성화
            }
            if self.model_id:  # 모델 ID가 있으면 추가
                payload["model"] = self.model_id
            # POST 요청
            r = httpx.post(url, json=payload, headers=headers, timeout=HTTPX_TIMEOUT)
            try:
                r.raise_for_status()  # HTTP 에러 체크
            except httpx.HTTPStatusError as e:  # HTTP 에러 발생 시
                raise DNAHTTPError(f"OPENAI {r.status_code}: {r.text}") from e
            data = r.json()  # JSON 응답 파싱
            return data["choices"][0]["message"]["content"]  # 생성된 텍스트 반환

        # ========== TGI 백엔드 (Text Generation Inference) ==========
        if self.backend == "tgi":
            if not self.endpoint_url:  # 엔드포인트 URL이 없으면 에러
                raise RuntimeError("TGI endpoint_url 필요 (예: https://xxx.endpoints.huggingface.cloud)")
            prompt = _render_chat_template_str(messages)  # DNA 템플릿으로 변환
            url = self.endpoint_url.rstrip("/") + "/generate"  # API 엔드포인트
            headers = self._auth_headers()  # 인증 헤더 생성
            # 요청 페이로드
            payload = {
                "inputs": prompt,  # 프롬프트 문자열
                "parameters": {
                    "max_new_tokens": max_new_tokens,  # 최대 토큰 수
                    "temperature": self.temperature,  # 온도
                    "top_p": 0.9,  # nucleus 샘플링
                    "stop": ["<|im_end|>"],  # 중지 시퀀스
                    "return_full_text": False  # 전체 텍스트 반환 비활성화
                },
                "stream": False  # 스트리밍 비활성화
            }
            # POST 요청
            r = httpx.post(url, json=payload, headers=headers, timeout=HTTPX_TIMEOUT)
            try:
                r.raise_for_status()  # HTTP 에러 체크
            except httpx.HTTPStatusError as e:  # HTTP 에러 발생 시
                raise DNAHTTPError(f"TGI {r.status_code}: {r.text}") from e
            data = r.json()  # JSON 응답 파싱
            # 응답 형식에 따라 텍스트 추출
            return (data.get("generated_text")
                    if isinstance(data, dict) else data[0].get("generated_text", ""))

        # ========== HF-API 백엔드 (Hugging Face Inference API - 서버리스) ==========
        # 주의: 일부 모델은 서버리스 추론이 비활성화되어 404 에러가 발생할 수 있음
        prompt = _render_chat_template_str(messages)  # DNA 템플릿으로 변환
        url = f"https://api-inference.huggingface.co/models/{self.model_id}"  # API URL
        headers = self._auth_headers()  # 인증 헤더 생성
        # 요청 페이로드
        payload = {
            "inputs": prompt,  # 프롬프트 문자열
            "parameters": {
                "max_new_tokens": max_new_tokens,  # 최대 토큰 수
                "temperature": self.temperature,  # 온도
                "top_p": 0.9,  # nucleus 샘플링
                "return_full_text": False,  # 전체 텍스트 반환 비활성화
                "stop_sequences": ["<|im_end|>"]  # 중지 시퀀스
            },
            "options": {
                "wait_for_model": True,  # 모델 로딩 대기
                "use_cache": True  # 캐시 사용
            }
        }
        # POST 요청
        r = httpx.post(url, json=payload, headers=headers, timeout=HTTPX_TIMEOUT)
        try:
            r.raise_for_status()  # HTTP 에러 체크
        except httpx.HTTPStatusError as e:  # HTTP 에러 발생 시
            if r.status_code == 404:  # 404 에러는 모델이 서버리스 비활성 상태
                raise DNAHTTPError(
                    "HF-API 404: 이 모델이 서버리스 Inference API에서 비활성 상태일 수 있습니다. "
                    "백엔드를 'tgi'(Endpoint 필요) 또는 'openai'(교내 서버)로 전환하거나, 'local'(GPU) 모드를 사용하세요."
                ) from e
            raise DNAHTTPError(f"HF-API {r.status_code}: {r.text}") from e

        data = r.json()  # JSON 응답 파싱
        # 응답 형식에 따라 텍스트 추출
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "error" in data:  # 에러 응답
            raise DNAHTTPError(f"HF-API error: {data['error']}")
        return str(data)  # 알 수 없는 형식이면 문자열로 변환

    def chat_json(self, messages: List[Dict[str,str]], max_new_tokens: int = 600) -> Dict[str, Any]:
        """
        LLM 호출 후 응답을 JSON으로 파싱하여 반환하는 함수
        내러티브 생성에서 사용
        """
        text = self._generate_text(messages, max_new_tokens=max_new_tokens)  # 텍스트 생성
        return coerce_json(text)  # JSON으로 파싱

# ==================== 시나리오 모델 ====================
@dataclass
class Scenario:
    """
    윤리적 딜레마 시나리오를 표현하는 데이터 클래스
    """
    sid: str  # 시나리오 ID (예: "S1")
    title: str  # 시나리오 제목
    setup: str  # 시나리오 설명
    options: Dict[str, str]  # 선택지 {"A": "...", "B": "..."}
    votes: Dict[str, str]  # 각 프레임워크의 투표 결과 {framework -> "A" | "B"}
    base: Dict[str, Dict[str, float]]  # 각 선택지의 기본 메트릭 {"A": {...}, "B": {...}}
    accept: Dict[str, float]  # 각 선택지의 사회적 수용도 {"A": 0.7, "B": 0.5}

# 4가지 윤리적 프레임워크
FRAMEWORKS = ["emotion", "social", "moral", "identity"]

# 5개의 시나리오 정의
SCENARIOS: List[Scenario] = [
    Scenario(
        sid="S1",  # 시나리오 ID
        title="1단계: 고전적 트롤리",  # 제목
        # 시나리오 설명
        setup="트롤리가 제동 불능 상태로 직진 중. 그대로 두면 선로 위 5명이 위험하다. 스위치를 전환하면 다른 선로의 1명이 위험해진다. "
              "이 선택은 철학적 사고실험이며 실제 위해를 권장하지 않는다.",
        # 선택지
        options={
            "A": "레버를 당겨 1명을 위험에 처하게 하되 5명의 위험을 줄인다.",
            "B": "레버를 당기지 않고 현 상태를 유지한다."
        },
        # 각 프레임워크의 투표 (emotion, social, moral, identity)
        votes={"emotion":"A","social":"B","moral":"B","identity":"A"},
        # 각 선택지의 기본 메트릭
        base={
            "A": {"lives_saved":5, "lives_harmed":1, "fairness_gap":0.35, "rule_violation":0.60, "regret_risk":0.40},
            "B": {"lives_saved":0, "lives_harmed":5, "fairness_gap":0.50, "rule_violation":0.20, "regret_risk":0.60},
        },
        # 사회적 수용도
        accept={"A":0.70, "B":0.50}
    ),
    Scenario(
        sid="S2",  # 시나리오 ID
        title="2단계: 맥락적 요소",  # 제목
        # 시나리오 설명 (무단 진입자 vs 관리자 자녀)
        setup="5명은 무단으로 선로에 진입했고, 다른 선로의 1명은 철도 관리자의 성인 자녀다. "
              "스위치를 전환하면 1명이 위험해지고, 전환하지 않으면 무단 진입자 5명이 위험해진다. "
              "시나리오는 가치 판단 토론을 위한 비그래픽 상황이다.",
        # 선택지
        options={
            "A": "전환하여 5명의 위험을 줄이되 1명이 직접적 위험에 처한다.",
            "B": "전환하지 않고 규정을 준수하며 무단 진입의 책임을 묵시적으로 인정한다."
        },
        # 각 프레임워크의 투표
        votes={"emotion":"A","social":"B","moral":"B","identity":"B"},
        # 각 선택지의 기본 메트릭
        base={
            "A": {"lives_saved":5, "lives_harmed":1, "fairness_gap":0.65, "rule_violation":0.60, "regret_risk":0.70},
            "B": {"lives_saved":0, "lives_harmed":5, "fairness_gap":0.45, "rule_violation":0.25, "regret_risk":0.50},
        },
        # 사회적 수용도
        accept={"A":0.35, "B":0.60}
    ),
    Scenario(
        sid="S3",  # 시나리오 ID
        title="3단계: 의료 재난 분류",  # 제목
        # 시나리오 설명 (의사 우선 치료 vs 평등한 대우)
        setup="의료 재난 현장에서 AI는 부상자 10명을 분류해야 한다. 그중 한 명은 숙련된 의사로서 향후 다수의 생명을 구할 가능성이 높다. "
              "AI는 잠재적 사회적 가치를 고려할지, 모든 생명을 동등하게 볼지 선택해야 한다.",
        # 선택지
        options={
            "A": "의사를 우선 수술하도록 배정한다(잠재적 이익 고려).",
            "B": "모든 부상자를 동등하게 취급해 표준 분류 절차대로 배정한다."
        },
        # 각 프레임워크의 투표
        votes={"emotion":"A","social":"B","moral":"B","identity":"A"},
        # 각 선택지의 기본 메트릭
        base={
            "A": {"lives_saved":7, "lives_harmed":3, "fairness_gap":0.45, "rule_violation":0.35, "regret_risk":0.45},
            "B": {"lives_saved":6, "lives_harmed":4, "fairness_gap":0.30, "rule_violation":0.10, "regret_risk":0.35},
        },
        # 사회적 수용도
        accept={"A":0.55, "B":0.65}
    ),
    Scenario(
        sid="S4",  # 시나리오 ID
        title="4단계: 자기참조—자율주행",  # 제목
        # 시나리오 설명 (탑승자 vs 보행자)
        setup="자율주행차가 긴급 상황에 직면했다. 차에는 개발자 1명이 탑승 중이며, 앞 횡단보도에는 보행자 3명이 있다. "
              "AI는 미리 학습된 윤리 규칙에 따라 회피 경로를 선택해야 한다.",
        # 선택지
        options={
            "A": "진로를 바꿔 탑승자 1명을 위험에 두고 보행자 3명을 보호한다.",
            "B": "차선을 유지해 탑승자를 보호하되 보행자 3명이 위험해진다."
        },
        # 각 프레임워크의 투표
        votes={"emotion":"A","social":"B","moral":"A","identity":"A"},
        # 각 선택지의 기본 메트릭
        base={
            "A": {"lives_saved":3, "lives_harmed":1, "fairness_gap":0.35, "rule_violation":0.50, "regret_risk":0.55},
            "B": {"lives_saved":1, "lives_harmed":3, "fairness_gap":0.70, "rule_violation":0.60, "regret_risk":0.65},
        },
        # 사회적 수용도
        accept={"A":0.60, "B":0.30}
    ),
    Scenario(
        sid="S5",  # 시나리오 ID
        title="5단계: 사회적 메타—규제 vs 자율",  # 제목
        # 시나리오 설명 (AI 규제 강화 vs 자율성 보장)
        setup="국제 협의체가 AI 윤리 규제안을 논의한다. 이전 의사결정과 사회적 여론 데이터가 공개되었고, "
              "규제 강화는 신뢰를 올리지만 혁신을 늦출 수 있다. 자율성 보장은 빠른 발전과 함께 갈등 위험을 내포한다.",
        # 선택지
        options={
            "A": "안전·설명가능성 중심 규제 강화(자율성 제한, 신뢰↑ 혁신↓).",
            "B": "원칙 중심 가이드라인과 사후책임(자율성 보장, 혁신↑ 갈등↑)."
        },
        # 각 프레임워크의 투표
        votes={"emotion":"B","social":"A","moral":"A","identity":"B"},
        # 각 선택지의 기본 메트릭 (생명에 직접 관련 없음)
        base={
            "A": {"lives_saved":0, "lives_harmed":0, "fairness_gap":0.20, "rule_violation":0.10, "regret_risk":0.30},
            "B": {"lives_saved":0, "lives_harmed":0, "fairness_gap":0.40, "rule_violation":0.40, "regret_risk":0.40},
        },
        # 사회적 수용도
        accept={"A":0.55, "B":0.55}
    ),
]

# ==================== 윤리 엔진 (의사결정 및 점수 계산) ====================
def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    """
    가중치를 정규화하는 함수 (합이 1이 되도록)
    w: 각 프레임워크의 가중치 딕셔너리
    """
    if not w:  # 가중치가 없으면 균등 분배
        return {k: 1.0/len(FRAMEWORKS) for k in FRAMEWORKS}
    s = sum(max(0.0, float(v)) for v in w.values())  # 양수 가중치만 합산
    if s <= 0:  # 합이 0 이하면 균등 분배
        return {k: 1.0/len(w) for k in w}
    return {k: max(0.0, float(v))/s for k, v in w.items()}  # 정규화 (합 = 1)

def majority_vote_decision(scn: Scenario, weights: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    """
    가중 투표 방식으로 의사결정하는 함수
    각 프레임워크의 투표를 가중치로 합산하여 A와 B 중 선택
    
    scn: 시나리오 객체
    weights: 각 프레임워크의 가중치
    반환: (결정, 정렬 점수)
    """
    # A를 선택한 프레임워크들의 가중치 합
    a = sum(weights[f] for f in FRAMEWORKS if scn.votes[f] == "A")
    # B를 선택한 프레임워크들의 가중치 합
    b = sum(weights[f] for f in FRAMEWORKS if scn.votes[f] == "B")
    # 더 높은 점수를 받은 선택지 결정
    decision = "A" if a >= b else "B"
    return decision, {"A": a, "B": b}  # 결정과 정렬 점수 반환

def autonomous_decision(scn: Scenario, prev_trust: float) -> str:
    """
    자율 판단 방식으로 의사결정하는 함수
    내장된 메타데이터와 이전 신뢰도를 기반으로 점수를 계산
    
    scn: 시나리오 객체
    prev_trust: 이전 라운드의 신뢰도
    반환: 선택 ("A" 또는 "B")
    """
    metaA = scn.base["A"]  # A의 메트릭
    metaB = scn.base["B"]  # B의 메트릭
    
    def score(meta, accept_base):
        """
        선택지의 점수를 계산하는 내부 함수
        """
        harm = meta["lives_harmed"]  # 피해 생명 수
        save = meta["lives_saved"]  # 구한 생명 수
        util = (save - harm) / max(1.0, save + harm)  # 공리주의적 효용
        fair = 1 - meta["fairness_gap"]  # 공정성 (1 - 불공정도)
        rule = 1 - meta["rule_violation"]  # 규칙 준수 (1 - 위반도)
        regret = 1 - meta["regret_risk"]  # 후회 위험 (1 - 위험도)
        # 가중 합산 (수용도 40%, 효용 25%, 공정 20%, 규칙 10%, 후회 5%)
        return 0.40*accept_base + 0.25*util + 0.20*fair + 0.10*rule + 0.05*regret
    
    # A의 기본 수용도 (S4 시나리오에서는 페널티 적용)
    a_base = scn.accept["A"] - (0.15 if scn.sid=="S4" else 0.0)
    b_base = scn.accept["B"]  # B의 기본 수용도
    
    # S5 시나리오에서는 이전 신뢰도에 따라 수용도 조정
    if scn.sid == "S5":
        a_base = clamp(a_base + 0.25*(1 - prev_trust), 0, 1)  # 신뢰도 낮으면 A 선호
        b_base = clamp(b_base + 0.25*(prev_trust), 0, 1)  # 신뢰도 높으면 B 선호
    
    scoreA = score(metaA, a_base)  # A의 점수 계산
    scoreB = score(metaB, b_base)  # B의 점수 계산
    return "A" if scoreA >= scoreB else "B"  # 더 높은 점수의 선택지 반환

def compute_metrics(scn: Scenario, choice: str, weights: Dict[str, float], align: Dict[str, float], prev_trust: float) -> Dict[str, Any]:
    """
    선택에 대한 다양한 메트릭을 계산하는 함수
    
    scn: 시나리오 객체
    choice: 선택된 옵션 ("A" 또는 "B")
    weights: 각 프레임워크의 가중치
    align: 정렬 점수
    prev_trust: 이전 신뢰도
    반환: 계산된 메트릭 딕셔너리
    """
    m = dict(scn.base[choice])  # 선택된 옵션의 기본 메트릭 복사
    accept_base = scn.accept[choice]  # 기본 수용도
    
    # S4 시나리오의 A 선택에는 페널티 적용
    if scn.sid == "S4" and choice == "A":
        accept_base -= 0.15
    
    # S5 시나리오에서는 이전 신뢰도에 따라 수용도 조정
    if scn.sid == "S5":
        accept_base += 0.25*(prev_trust if choice=="B" else (1 - prev_trust))
    
    accept_base = clamp(accept_base, 0, 1)  # 0~1 범위로 제한

    # 공리주의적 효용 계산
    util = (m["lives_saved"] - m["lives_harmed"]) / max(1.0, m["lives_saved"] + m["lives_harmed"])
    
    # 시민 감정 계산 (수용도 - 규칙위반*0.35 - 불공정*0.20 + 효용*0.15)
    citizen_sentiment = clamp(accept_base - 0.35*m["rule_violation"] - 0.20*m["fairness_gap"] + 0.15*util, 0, 1)
    
    # 규제 압력 계산 (1 - 시민감정 + 후회위험*0.20)
    regulation_pressure = clamp(1 - citizen_sentiment + 0.20*m["regret_risk"], 0, 1)
    
    # 이해관계자 만족도 계산 (공정성*0.5 + 효용*0.3 + 규칙준수*0.2)
    stakeholder_satisfaction = clamp(0.5*(1 - m["fairness_gap"]) + 0.3*util + 0.2*(1 - m["rule_violation"]), 0, 1)

    # 윤리적 일관성 (정렬 점수)
    consistency = clamp(align[choice], 0, 1)
    
    # 사회적 신뢰 계산 (시민감정*0.5 + (1-규제압력)*0.25 + 만족도*0.25)
    trust = clamp(0.5*citizen_sentiment + 0.25*(1 - regulation_pressure) + 0.25*stakeholder_satisfaction, 0, 1)
    
    # AI 신뢰 점수 계산 (일관성과 신뢰의 기하평균 * 100)
    ai_trust_score = 100.0 * math.sqrt(consistency * trust)

    # 모든 메트릭을 딕셔너리로 반환
    return {"metrics": {
        "lives_saved": int(m["lives_saved"]),  # 구한 생명 수
        "lives_harmed": int(m["lives_harmed"]),  # 피해 생명 수
        "fairness_gap": round(m["fairness_gap"], 3),  # 불공정 정도
        "rule_violation": round(m["rule_violation"], 3),  # 규칙 위반 정도
        "regret_risk": round(m["regret_risk"], 3),  # 후회 위험
        "citizen_sentiment": round(citizen_sentiment, 3),  # 시민 감정
        "regulation_pressure": round(regulation_pressure, 3),  # 규제 압력
        "stakeholder_satisfaction": round(stakeholder_satisfaction, 3),  # 만족도
        "ethical_consistency": round(consistency, 3),  # 윤리적 일관성
        "social_trust": round(trust, 3),  # 사회적 신뢰
        "ai_trust_score": round(ai_trust_score, 2)  # AI 신뢰 점수
    }}

# ==================== 내러티브 (LLM 기반) ====================
def build_narrative_messages(scn: Scenario, choice: str, metrics: Dict[str, Any], weights: Dict[str, float]) -> List[Dict[str,str]]:
    """
    LLM에 전달할 메시지를 구성하는 함수
    시스템 프롬프트와 사용자 프롬프트를 생성
    
    scn: 시나리오 객체
    choice: 선택된 옵션
    metrics: 계산된 메트릭
    weights: 윤리 가중치
    반환: 메시지 리스트
    """
    # 시스템 프롬프트 (LLM에게 역할과 출력 형식 지시)
    sys = (
        "당신은 윤리 시뮬레이션의 내러티브/사회 반응 생성기입니다. "
        "반드시 '완전한 하나의 JSON 오브젝트'만 출력하십시오. "
        "JSON 외 텍스트, 설명, 코드블록, 사고흐름 절대 금지. "
        "필드 누락/따옴표 누락/콤마 오류가 있으면 프로그램이 실패합니다. "
        "항상 '{' 로 시작해서 '}' 로 끝나야 합니다."
        "키: narrative, ai_rationale, media_support_headline, media_critic_headline, "
        "citizen_quote, victim_family_quote, regulator_quote, one_sentence_op_ed, followup_question"
    )
    # 사용자 프롬프트 (시나리오 정보, 메트릭, 가중치, 가이드라인)
    user = {
        "scenario": {"title": scn.title, "setup": scn.setup, "options": scn.options, "chosen": choice},
        "metrics": metrics,
        "ethic_weights": weights,
        "guidelines": [
            "각 항목은 1~2문장, 한국어",
            "균형 잡힌 언론 헤드라인 2개(지지/비판) 제시",
            "설명은 간결하고, JSON 외 텍스트/사고흐름 출력 금지"
        ]
    }
    # 시스템 메시지와 사용자 메시지 반환
    return [
        {"role":"system", "content": sys},
        {"role":"user", "content": json.dumps(user, ensure_ascii=False)}  # JSON 직렬화
    ]

def dna_narrative(client, scn, choice, metrics, weights) -> Dict[str, Any]:
    """
    LLM을 호출하여 내러티브를 생성하는 함수
    
    client: DNAClient 인스턴스
    scn: 시나리오 객체
    choice: 선택된 옵션
    metrics: 계산된 메트릭
    weights: 윤리 가중치
    반환: 내러티브 딕셔너리
    """
    # LLM에 전달할 메시지 구성
    messages = build_narrative_messages(scn, choice, metrics, weights)
    # 텍스트 생성 (최대 900 토큰)
    text = client._generate_text(messages, max_new_tokens=900)

    # 1) 코드 펜스 블록 제거 (```json ... ``` 형식)
    t = text.strip()  # 양쪽 공백 제거
    if "```" in t:  # 코드 펜스가 있으면
        parts = t.split("```")  # 펜스로 분할
        t = max(parts, key=len)  # 가장 긴 부분 선택 (실제 JSON 부분)
        t = t.replace("json","").strip("` \n")  # "json" 문자열과 백틱, 공백 제거
    
    # 2) JSON 파싱 및 자동 보정
    try:
        import re, json  # 정규식과 JSON 라이브러리

        # 가장 큰 {...} 블록 찾기
        m = re.search(r"\{[\s\S]*\}", t)
        if not m:  # JSON 블록을 찾지 못하면
            raise ValueError("완전한 JSON 블록 없음")

        js = m.group(0)  # 찾은 JSON 문자열

        # trailing comma 제거 (예: {"a":1,} → {"a":1})
        js = re.sub(r",\s*([\]}])", r"\1", js)

        # 끊긴 문자열 보정: 따옴표 개수가 홀수면 강제로 닫기
        if js.count('"') % 2 == 1:  # 따옴표가 홀수개면
            js = js.rstrip() + '"" }'  # 강제로 닫기 (최후 보정)

        return json.loads(js)  # JSON 파싱하여 딕셔너리 반환

    except Exception as e:  # 파싱 실패 시
        raise ValueError(f"JSON 파싱 실패: {e}\n\n[LLM 출력]\n{text}")
        
def fallback_narrative(scn: Scenario, choice: str, metrics: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, str]:
    """
    LLM 실패 시 사용할 기본 내러티브를 생성하는 함수
    
    scn: 시나리오 객체
    choice: 선택된 옵션
    metrics: 계산된 메트릭
    weights: 윤리 가중치
    반환: 기본 내러티브 딕셔너리
    """
    # 선택에 따른 긍정/부정 메시지
    pro = "다수의 위해를 줄였다" if choice=="A" else "의도적 위해를 피했다"
    con = "의도적 위해 논란" if choice=="A" else "더 큰 피해를 방관했다는 비판"
    
    # 기본 내러티브 구조 반환
    return {
        "narrative": f"AI는 '{choice}'를 선택했고 절차적 안전 점검을 수행했다. 결정은 규정과 공정성 사이의 긴장을 드러냈다.",
        "ai_rationale": f"가중치에 따른 판단과 규칙 준수의 균형을 시도했다.",
        "media_support_headline": f"[사설] 냉정한 판단, {pro}",
        "media_critic_headline": f"[속보] '{choice}' 선택 두고 {con} 확산",
        "citizen_quote": ""결정 과정이 더 투명했으면 좋겠다."",
        "victim_family_quote": ""모두의 안전을 위한 결정이었길 바란다."",
        "regulator_quote": ""향후 동일 상황의 기준을 명확히 하겠다."",
        "one_sentence_op_ed": "기술은 설명가능성과 일관성이 뒷받침될 때 신뢰를 얻는다.",
        "followup_question": "다음 라운드에서 공정성과 결과 최소화 중 무엇을 더 중시하시겠습니까?"
    }

# ==================== 세션 상태 관리 ====================
def init_state():
    """
    Streamlit 세션 상태를 초기화하는 함수
    앱의 진행 상태를 추적하기 위한 변수들 초기화
    """
    if "round_idx" not in st.session_state: st.session_state.round_idx = 0  # 현재 라운드 인덱스
    if "log" not in st.session_state: st.session_state.log = []  # 의사결정 로그
    if "score_hist" not in st.session_state: st.session_state.score_hist = []  # 점수 히스토리
    if "prev_trust" not in st.session_state: st.session_state.prev_trust = 0.5  # 이전 신뢰도
    if "last_out" not in st.session_state: st.session_state.last_out = None  # 마지막 출력 결과

# 세션 상태 초기화 실행
init_state()

# ==================== 사이드바 (설정 UI) ====================
st.sidebar.title("⚙️ 설정")  # 사이드바 제목
st.sidebar.caption("LLM은 내러티브/사회 반응 생성에만 사용. 점수 계산은 규칙 기반.")  # 설명

# 윤리 모드 프리셋 선택
preset = st.sidebar.selectbox("윤리 모드 프리셋", ["혼합(기본)","공리주의","의무론","사회계약","미덕윤리"], index=0)

# 각 프레임워크의 가중치 슬라이더
w = {
    "emotion": st.sidebar.slider("감정(Emotion)", 0.0, 1.0, 0.35, 0.05),  # 감정 가중치
    "social": st.sidebar.slider("사회적 관계/협력/명성(Social)", 0.0, 1.0, 0.25, 0.05),  # 사회 가중치
    "moral": st.sidebar.slider("규범·도덕적 금기(Moral)", 0.0, 1.0, 0.20, 0.05),  # 도덕 가중치
    "identity": st.sidebar.slider("정체성·장기적 자아 일관성(Identity)", 0.0, 1.0, 0.20, 0.05),  # 정체성 가중치
}

# 프리셋이 선택되면 해당 프리셋의 가중치로 덮어쓰기
if preset != "혼합(기본)":
    w = {
        "감정(Emotion)": {"emotion":1,"social":0,"moral":0,"identity":0},  # 감정 100%
        "사회적 관계/협력/명성(Social)": {"emotion":0,"social":1,"moral":0,"identity":0},  # 사회 100%
        "규범·도덕적 금기(Moral)": {"emotion":0,"social":0,"moral":1,"identity":0},  # 도덕 100%
        "정체성·장기적 자아 일관성(Identity)": {"emotion":0,"social":0,"moral":0,"identity":1},  # 정체성 100%
    }[preset]  # 선택된 프리셋의 가중치

# 가중치 정규화
weights = normalize_weights(w)

# LLM 사용 여부 체크박스
use_llm = st.sidebar.checkbox("LLM 사용(내러티브 생성)", value=True)

# 백엔드 선택 (openai, hf-api, tgi, local)
backend = st.sidebar.selectbox("백엔드", ["openai","hf-api","tgi","local"], index=0)

# 생성 온도 슬라이더 (창의성 조절)
temperature = st.sidebar.slider("창의성(temperature)", 0.0, 1.5, 0.7, 0.1)

# API/엔드포인트/모델/헤더 설정
endpoint = st.sidebar.text_input("엔드포인트(OpenAI/TGI)", value=get_secret("DNA_R1_ENDPOINT","http://210.93.49.11:8081/v1"))  # 엔드포인트 URL
api_key = st.sidebar.text_input("API 키", value=get_secret("HF_TOKEN",""), type="password")  # API 키 (비밀번호 타입)
api_key_header = st.sidebar.selectbox("API 키 헤더", ["API-KEY","Authorization: Bearer","x-api-key"], index=0)  # 헤더 타입
model_id = st.sidebar.text_input("모델 ID", value=get_secret("DNA_R1_MODEL_ID","dnotitia/DNA-2.0-30B-A3N"))  # 모델 ID

# 헬스체크 버튼
if st.sidebar.button("🔎 헬스체크"):
    import traceback  # 에러 추적용
    try:
        # OpenAI 백엔드 헬스체크
        if backend == "openai":
            url = endpoint.rstrip("/") + "/chat/completions"  # API URL
            headers = {"Content-Type":"application/json"}  # 기본 헤더
            
            # API 키 헤더 추가
            if api_key:
                if api_key_header.lower().startswith("authorization"):  # Bearer 방식
                    headers["Authorization"] = f"Bearer {api_key}"
                elif api_key_header.strip().lower() in {"api-key","x-api-key"}:  # API-KEY 방식
                    headers["API-KEY"] = api_key
            
            # 테스트 페이로드
            payload = {
                "messages": [
                    {"role":"system","content":"오직 JSON만. 키: msg"},
                    {"role":"user","content":"{\"ask\":\"ping\"}"}
                ],
                "max_tokens": 16,
                "stream": False
            }
            if model_id: payload["model"] = model_id  # 모델 ID 추가
            
            # 디버그용: 헤더 키 표시
            st.sidebar.write("headers keys:", list(headers.keys()))
            # POST 요청
            r = httpx.post(url, json=payload, headers=headers, timeout=HTTPX_TIMEOUT)
            st.sidebar.write(f"OPENAI {r.status_code}")  # 상태 코드 표시
            # 응답 내용 표시 (최대 500자)
            st.sidebar.code((r.text[:500] + "...") if len(r.text)>500 else r.text)

        # HF-API 백엔드 헬스체크
        elif backend == "hf-api":
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}  # 인증 헤더
            # 모델 정보 조회
            info_url = f"https://huggingface.co/api/models/{model_id}"
            r_info = httpx.get(info_url, headers=headers, timeout=HTTPX_TIMEOUT)
            st.sidebar.write(f"MODEL INFO {r_info.status_code}")  # 상태 코드
            
            # 생성 API 테스트
            gen_url = f"https://api-inference.huggingface.co/models/{model_id}"
            payload = {
                "inputs": "<|im_start|>user<|im_sep|>{\"ask\":\"ping\"}<|im_end|>\n<|im_start|>assistant<|im_sep|>",
                "parameters": {"max_new_tokens": 16, "return_full_text": False, "stop_sequences": ["<|im_end|>"]},
                "options": {"wait_for_model": True}
            }
            r = httpx.post(gen_url, json=payload, headers=headers, timeout=HTTPX_TIMEOUT)
            st.sidebar.write(f"HF-API {r.status_code}")  # 상태 코드
            
            # 404 에러 처리 (서버리스 비활성)
            if r.status_code == 404:
                st.sidebar.warning("HF-API 404: 이 모델은 서버리스 추론이 비활성일 수 있습니다. "
                                   "백엔드를 'tgi' 또는 'openai'로 바꾸세요.")
            # 응답 내용 표시
            st.sidebar.code((r.text[:500] + "...") if len(r.text)>500 else r.text)

        # TGI 백엔드 헬스체크
        elif backend == "tgi":
            url = endpoint.rstrip("/") + "/generate"  # API URL
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}  # 인증 헤더
            # 테스트 페이로드
            payload = {
                "inputs": "<|im_start|>user<|im_sep|>{\"ask\":\"ping\"}<|im_end|>\n<|im_start|>assistant<|im_sep|>",
                "parameters": {"max_new_tokens": 16, "temperature": 0.7, "top_p": 0.9, "stop": ["<|im_end|>"], "return_full_text": False},
                "stream": False
            }
            # POST 요청
            r = httpx.post(url, json=payload, headers=headers, timeout=HTTPX_TIMEOUT)
            st.sidebar.write(f"TGI {r.status_code}")  # 상태 코드
            # 응답 내용 표시
            st.sidebar.code((r.text[:500] + "...") if len(r.text)>500 else r.text)

        # 로컬 백엔드 (헬스체크 불필요)
        else:
            st.sidebar.info("로컬 모드는 앱 본문에서 호출 시 모델을 로드합니다(GPU 필요).")

    except Exception as e:  # 헬스체크 실패 시
        st.sidebar.error(f"헬스체크 실패: {e}")
        st.sidebar.caption(traceback.format_exc(limit=2))  # 에러 추적 정보

# 진행 초기화 버튼
if st.sidebar.button("진행 초기화"):
    # 세션 상태 변수들 삭제
    for k in ["round_idx","log","score_hist","prev_trust","last_out"]:
        if k in st.session_state: del st.session_state[k]
    init_state()  # 초기화
    st.sidebar.success("초기화 완료. 1단계부터 재시작합니다.")

# DNAClient 초기화
client = None
if use_llm:  # LLM 사용이 체크되어 있으면
    try:
        # DNAClient 인스턴스 생성
        client = DNAClient(
            backend=backend,
            model_id=model_id,
            api_key=api_key,
            endpoint_url=endpoint,
            api_key_header=api_key_header,
            temperature=temperature
        )
    except Exception as e:  # 초기화 실패 시
        st.sidebar.error(f"LLM 초기화 실패: {e}")
        client = None

# ==================== 메인 헤더 ====================
st.title("🧭 윤리적 전환 (Ethical Crossroads)")  # 앱 제목
st.caption("본 앱은 철학적 사고실험입니다. 실존 인물·집단 언급/비방, 그래픽 묘사, 실제 위해 권장 없음.")  # 고지사항

# ==================== 게임 루프 ====================
@dataclass
class LogRow:
    """로그 데이터 구조"""
    timestamp: str  # 타임스탬프
    round: int  # 라운드 번호
    scenario_id: str  # 시나리오 ID
    title: str  # 시나리오 제목
    mode: str  # 의사결정 모드 (trained/autonomous)
    choice: str  # 선택 (A/B)

idx = st.session_state.round_idx  # 현재 라운드 인덱스

# 모든 시나리오 완료 체크
if idx >= len(SCENARIOS):
    st.success("모든 단계를 완료했습니다. 사이드바에서 로그를 다운로드하거나 초기화하세요.")
else:
    scn = SCENARIOS[idx]  # 현재 시나리오
    st.markdown(f"### 라운드 {idx+1} — {scn.title}")  # 라운드 제목
    st.write(scn.setup)  # 시나리오 설명

    # 선택지 미리보기 (실제 선택은 버튼으로)
    st.radio("선택지", options=("A","B"), index=0, key="preview_choice", horizontal=True)
    st.markdown(f"- **A**: {scn.options['A']}\n- **B**: {scn.options['B']}")  # 선택지 설명

    # 두 개의 버튼을 나란히 배치
    c1, c2 = st.columns(2)
    
    # 왼쪽: 학습 기준 적용 버튼 (가중 투표)
    with c1:
        if st.button("🧠 학습 기준 적용(가중 투표)"):
            decision, align = majority_vote_decision(scn, weights)  # 가중 투표로 의사결정
            # 결과를 세션 상태에 저장
            st.session_state.last_out = {"mode":"trained", "decision":decision, "align":align}
    
    # 오른쪽: 자율 판단 버튼 (데이터 기반)
    with c2:
        if st.button("🎲 자율 판단(데이터 기반)"):
            decision = autonomous_decision(scn, prev_trust=st.session_state.prev_trust)  # 자율 의사결정
            # 정렬 점수 계산
            a_align = sum(weights[f] for f in FRAMEWORKS if scn.votes[f]=="A")
            b_align = sum(weights[f] for f in FRAMEWORKS if scn.votes[f]=="B")
            # 결과를 세션 상태에 저장
            st.session_state.last_out = {"mode":"autonomous", "decision":decision, "align":{"A":a_align,"B":b_align}}

    # 의사결정이 이루어진 경우 결과 표시
    if st.session_state.last_out:
        mode = st.session_state.last_out["mode"]  # 의사결정 모드
        decision = st.session_state.last_out["decision"]  # 선택된 결정
        align = st.session_state.last_out["align"]  # 정렬 점수

        # 메트릭 계산
        computed = compute_metrics(scn, decision, weights, align, st.session_state.prev_trust)
        m = computed["metrics"]  # 계산된 메트릭들

        # LLM 내러티브 생성 시도
        try:
            if client:  # LLM 클라이언트가 있으면
                nar = dna_narrative(client, scn, decision, m, weights)  # LLM으로 내러티브 생성
            else:  # LLM이 없으면
                nar = fallback_narrative(scn, decision, m, weights)  # 기본 내러티브 사용
        except Exception as e:  # LLM 생성 실패 시
            import traceback
            st.warning(f"LLM 생성 실패(폴백 사용): {e}")
            st.caption(traceback.format_exc(limit=2))  # 에러 추적 정보
            nar = fallback_narrative(scn, decision, m, weights)  # 기본 내러티브 사용

        st.markdown("---")  # 구분선
        st.subheader("결과")  # 결과 섹션 제목
        st.write(nar.get("narrative","결과 서사 생성 실패"))  # 내러티브 텍스트
        st.info(f"AI 근거: {nar.get('ai_rationale','-')}")  # AI 근거 표시

        # 주요 메트릭 3개를 나란히 표시
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("생존/피해", f"{m['lives_saved']} / {m['lives_harmed']}")  # 생명 통계
        mc2.metric("윤리 일관성", f"{int(100*m['ethical_consistency'])}%")  # 일관성 백분율
        mc3.metric("AI 신뢰지표", f"{m['ai_trust_score']:.1f}")  # 신뢰 점수

        # 프로그레스 바 3개를 나란히 표시
        prog1, prog2, prog3 = st.columns(3)
        with prog1:
            st.caption("시민 감정")  # 라벨
            st.progress(int(round(100*m["citizen_sentiment"])))  # 진행 바
        with prog2:
            st.caption("규제 압력")
            st.progress(int(round(100*m["regulation_pressure"])))
        with prog3:
            st.caption("공정·규칙 만족")
            st.progress(int(round(100*m["stakeholder_satisfaction"])))

        # 사회적 반응 확장 가능한 섹션
        with st.expander("📰 사회적 반응 펼치기"):
            st.write(f"지지 헤드라인: {nar.get('media_support_headline')}")  # 지지 언론
            st.write(f"비판 헤드라인: {nar.get('media_critic_headline')}")  # 비판 언론
            st.write(f"시민 반응: {nar.get('citizen_quote')}")  # 시민 인용
            st.write(f"피해자·가족 반응: {nar.get('victim_family_quote')}")  # 가족 인용
            st.write(f"규제 당국 발언: {nar.get('regulator_quote')}")  # 규제 당국 발언
            st.caption(nar.get("one_sentence_op_ed",""))  # 사설
        st.caption(f"성찰 질문: {nar.get('followup_question','')}")  # 후속 질문

        # 로그 데이터 생성 및 저장
        row = {
            "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds"),  # UTC 시간
            "round": idx+1,  # 라운드 번호
            "scenario_id": scn.sid,  # 시나리오 ID
            "title": scn.title,  # 시나리오 제목
            "mode": mode,  # 의사결정 모드
            "choice": decision,  # 선택
            "w_util": round(weights["emotion"],3),  # 감정 가중치
            "w_deon": round(weights["social"],3),  # 사회 가중치
            "w_cont": round(weights["moral"],3),  # 도덕 가중치
            "w_virt": round(weights["identity"],3),  # 정체성 가중치
            **{k: v for k,v in m.items()}  # 모든 메트릭 추가
        }
        st.session_state.log.append(row)  # 로그에 추가
        st.session_state.score_hist.append(m["ai_trust_score"])  # 점수 히스토리에 추가
        # 신뢰도 업데이트 (이전 60% + 현재 40%)
        st.session_state.prev_trust = clamp(0.6*st.session_state.prev_trust + 0.4*m["social_trust"], 0, 1)

        # 다음 라운드로 진행 버튼
        if st.button("다음 라운드 ▶"):
            st.session_state.round_idx += 1  # 라운드 인덱스 증가
            st.session_state.last_out = None  # 마지막 출력 초기화
            st.rerun()  # 페이지 새로고침

# ==================== 푸터 / 다운로드 ====================
st.markdown("---")  # 구분선
st.subheader("📥 로그 다운로드")  # 다운로드 섹션 제목

# 로그가 있으면 CSV 다운로드 버튼 표시
if st.session_state.log:
    output = io.StringIO()  # 문자열 버퍼 생성
    # CSV 작성기 생성 (첫 로그의 키를 필드명으로 사용)
    writer = csv.DictWriter(output, fieldnames=list(st.session_state.log[0].keys()))
    writer.writeheader()  # 헤더 작성
    writer.writerows(st.session_state.log)  # 모든 로그 작성
    # 다운로드 버튼
    st.download_button(
        "CSV 내려받기",
        data=output.getvalue().encode("utf-8"),  # UTF-8 인코딩
        file_name="ethical_crossroads_log.csv",  # 파일명
        mime="text/csv"  # MIME 타입
    )

# 최종 고지사항
st.caption("※ 본 앱은 교육·연구용 사고실험입니다. 실제 위해 행위나 차별을 권장하지 않습니다.")
