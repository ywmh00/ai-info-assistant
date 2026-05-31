"""
개인 맞춤형 AI 정보 비서 v4
============================
실행: streamlit run app.py
의존성: pip install streamlit==1.35.0 pandas scikit-learn
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────
# 1. 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI 정보 비서",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# 2. 커스텀 CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --bg:       #0d0f14;
    --surface:  #161923;
    --surface2: #1e2330;
    --border:   #2a303f;
    --text:     #e8eaf0;
    --muted:    #7a8099;
    --accent:   #5b8af5;
    --accent2:  #f5a623;
    --trust:    #4ecca3;
    --innov:    #f5627a;
    --radius:   12px;
}
html, body, [class*="css"] {
    font-family: -apple-system, 'Segoe UI', 'Malgun Gothic', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
.app-header {
    padding: 2rem 0 1.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.app-title {
    font-family: Georgia, 'Batang', serif;
    font-size: 2.1rem;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #e8eaf0 0%, #5b8af5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.2rem;
}
.app-subtitle { color: var(--muted); font-size: 0.82rem; letter-spacing: 0.05em; }
.control-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; color: var(--muted);
    font-weight: 500; font-size: 0.85rem;
    padding: 0.45rem 1rem; transition: all 0.2s;
}
.stTabs [aria-selected="true"] { background: var(--accent) !important; color: #fff !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }
.section-label {
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 0.75rem;
}
.section-label.trust { color: var(--trust); }
.section-label.innov { color: var(--innov); }

/* 통계 카드 */
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.3rem;
    text-align: center;
}
.stat-value {
    font-size: 2rem; font-weight: 800;
    line-height: 1.1; margin-bottom: 0.2rem;
}
.stat-label {
    font-size: 0.75rem; color: var(--muted);
    letter-spacing: 0.06em; text-transform: uppercase;
}
.stat-sub {
    font-size: 0.72rem; color: var(--muted);
    margin-top: 0.25rem;
}

/* 수평 바 차트 */
.hbar-row { display:flex; align-items:center; gap:0.6rem; margin-bottom:0.55rem; }
.hbar-label { font-size:0.78rem; color:var(--text); width:110px; flex-shrink:0; text-align:right; }
.hbar-bg { flex:1; height:10px; background:var(--surface2); border-radius:99px; overflow:hidden; }
.hbar-fill { height:100%; border-radius:99px; }
.hbar-val { font-size:0.75rem; font-weight:700; width:32px; flex-shrink:0; }

/* 일반 카드 */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    margin-bottom: 0.7rem;
    transition: border-color 0.2s, transform 0.15s;
}
.card:hover { border-color: var(--accent); transform: translateY(-1px); }
.card.trust-card { border-left: 3px solid var(--trust); }
.card.innov-card  { border-left: 3px solid var(--innov); }
.card.bkmk-card   { border-left: 3px solid var(--accent2); }
.card.is-read     { background: #12151c; opacity: 0.65; }
.read-badge {
    display: inline-block; border-radius: 20px;
    padding: 0.1rem 0.6rem; font-size: 0.68rem;
    font-weight: 700; letter-spacing: 0.03em;
}
.read-badge.unread {
    background: rgba(91,138,245,0.18); color: var(--accent);
    border: 1px solid rgba(91,138,245,0.35);
}
.read-badge.read {
    background: rgba(122,128,153,0.15); color: var(--muted);
    border: 1px solid rgba(122,128,153,0.25);
}
.card-title {
    font-family: Georgia, 'Batang', serif;
    font-size: 1rem; margin-bottom: 0.3rem; color: var(--text);
}
.card-meta {
    font-size: 0.75rem; color: var(--muted);
    margin-bottom: 0.55rem;
    display: flex; gap: 0.85rem; flex-wrap: wrap; align-items: center;
}
.card-summary {
    font-size: 0.82rem; color: #b0b5c8;
    line-height: 1.65; margin-bottom: 0.65rem;
}
.gauge-wrap { margin-bottom: 0.75rem; }
.gauge-row  { display:flex; align-items:center; gap:0.55rem; margin-bottom:0.32rem; }
.gauge-label { font-size:0.7rem; color:var(--muted); width:68px; flex-shrink:0; }
.gauge-bg { flex:1; height:6px; background:var(--surface2); border-radius:99px; overflow:hidden; }
.gauge-fill { height:100%; border-radius:99px; }
.gauge-val  { font-size:0.7rem; font-weight:700; width:26px; text-align:right; flex-shrink:0; }
.tag {
    display:inline-block; background:var(--surface2);
    border:1px solid var(--border); border-radius:20px;
    padding:0.1rem 0.52rem; font-size:0.68rem;
    color:var(--accent); margin-right:0.28rem; margin-bottom:0.28rem;
}
.cat-badge { display:inline-block; border-radius:6px; padding:0.1rem 0.5rem; font-size:0.68rem; font-weight:600; }
.cat-news  { background:rgba(91,138,245,0.15); color:#7aadff; }
.cat-paper { background:rgba(78,204,163,0.15); color:var(--trust); }
.cat-sns   { background:rgba(245,166,35,0.15); color:var(--accent2); }
.divider { border:none; border-top:1px solid var(--border); margin:1.3rem 0; }
.empty-state { text-align:center; padding:2.5rem 1rem; color:var(--muted); font-size:0.88rem; }
.stTextInput > div > div > input,
.stDateInput  > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
.stSlider [data-baseweb="slider"] { padding: 0; }
.stDataFrame { border-radius: var(--radius); overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 3. 세션 상태 초기화
# ──────────────────────────────────────────────
if "read_status" not in st.session_state:
    st.session_state.read_status = {}

# ──────────────────────────────────────────────
# 4. 샘플 데이터
# ──────────────────────────────────────────────
today = datetime.now().date()

raw = [
    {"date": today, "category": "뉴스",
     "title": "GPT-5 발표: 멀티모달 추론 능력 혁신",
     "source": "MIT Technology Review", "author_trust": 95, "innovation_score": 88,
     "summary": "OpenAI가 GPT-5를 공개하며 멀티모달 추론 능력을 대폭 강화했다.\n이미지·오디오·텍스트를 동시에 처리하는 새 아키텍처가 도입됐다.\n전문가들은 AGI 전환점이 될 수 있다고 평가한다.",
     "url": "https://www.technologyreview.com", "is_bookmarked": True, "tags": "#LLM #AGI #OpenAI"},

    {"date": today, "category": "뉴스",
     "title": "2025 글로벌 AI 스타트업 투자 트렌드 보고서",
     "source": "CB Insights", "author_trust": 91, "innovation_score": 62,
     "summary": "2025년 상반기 AI 스타트업 투자액이 전년 대비 34% 증가했다.\n헬스케어와 제조업 AI 솔루션이 가장 큰 투자 유입을 기록했다.\n시드 투자는 감소하고 시리즈 B 이상 대형 딜이 집중됐다.",
     "url": "https://www.cbinsights.com", "is_bookmarked": False, "tags": "#투자 #스타트업 #트렌드"},

    {"date": today, "category": "SNS/트렌드",
     "title": "소형 언어모델(SLM)의 역습: 엣지 AI 시대 개막",
     "source": "The Information", "author_trust": 78, "innovation_score": 96,
     "summary": "1B 파라미터 이하 소형 모델이 온디바이스 추론에서 GPT-4를 앞서는 사례가 등장했다.\n삼성·퀄컴이 전용 NPU 탑재를 확대하며 하드웨어 생태계가 성숙하고 있다.\n프라이버시와 지연 시간 측면에서 SLM의 경쟁력이 부각된다.",
     "url": "https://www.theinformation.com", "is_bookmarked": True, "tags": "#SLM #엣지AI #하드웨어"},

    {"date": today, "category": "논문",
     "title": "AI 규제 프레임워크 EU vs 미국 비교 분석",
     "source": "Brookings Institution", "author_trust": 93, "innovation_score": 45,
     "summary": "EU AI Act와 미국 행정명령 간 규제 철학의 근본적 차이가 드러났다.\n유럽은 위험 기반 사전 규제, 미국은 혁신 우선 자율 규제를 선호한다.\n글로벌 기업들은 이중 컴플라이언스 부담이 증가한다고 우려한다.",
     "url": "https://www.brookings.edu", "is_bookmarked": False, "tags": "#규제 #정책 #EU"},

    {"date": today, "category": "논문",
     "title": "멀티에이전트 협업 시스템의 창발적 행동 관찰",
     "source": "DeepMind Research", "author_trust": 96, "innovation_score": 94,
     "summary": "DeepMind 연구팀이 100개 이상 LLM 에이전트 협력 실험에서 예상치 못한 창발 행동을 발견했다.\n에이전트들이 명시적 설계 없이 자발적으로 역할 분업·품질 검토 프로세스를 형성했다.\n이 결과는 AI 집단 지성 연구의 새로운 방향을 제시한다.",
     "url": "https://deepmind.google/research", "is_bookmarked": True, "tags": "#멀티에이전트 #창발 #LLM"},

    {"date": today - timedelta(days=1), "category": "논문",
     "title": "Claude 4 Opus 벤치마크 심층 분석",
     "source": "Hugging Face Blog", "author_trust": 85, "innovation_score": 79,
     "summary": "Anthropic Claude 4 Opus가 MMLU·HumanEval 등 주요 벤치마크에서 최고 성능을 기록했다.\n특히 코드 생성과 수학 추론 영역에서 두드러진 향상이 확인됐다.\n연구자들은 안전성과 성능 균형이 이전 모델 대비 개선됐다고 평가한다.",
     "url": "https://huggingface.co/blog", "is_bookmarked": True, "tags": "#LLM #벤치마크 #Anthropic"},

    {"date": today - timedelta(days=1), "category": "뉴스",
     "title": "RAG 아키텍처 2.0: 그래프 기반 지식 검색",
     "source": "Towards Data Science", "author_trust": 72, "innovation_score": 91,
     "summary": "그래프 DB와 벡터 검색을 결합한 하이브리드 RAG가 주목받고 있다.\n엔티티 관계를 명시적으로 모델링해 다단계 추론 성능이 30% 향상됐다.\nNeo4j·Weaviate 연동 레퍼런스 구현이 GitHub에 공개됐다.",
     "url": "https://towardsdatascience.com", "is_bookmarked": True, "tags": "#RAG #그래프DB #검색"},

    {"date": today - timedelta(days=1), "category": "SNS/트렌드",
     "title": "X(트위터)에서 폭발한 'AI 슬로프' 논쟁",
     "source": "X Trending", "author_trust": 55, "innovation_score": 83,
     "summary": "AI 도구 과의존으로 사고력이 퇴화한다는 'AI 슬로프' 개념이 X에서 바이럴됐다.\n찬성 측은 계산기처럼 도구일 뿐이라고 반박하고, 반대 측은 메타인지 위협을 경고한다.\n교육계와 기업 HR 담당자들의 논의가 활발히 이어지고 있다.",
     "url": "https://x.com", "is_bookmarked": False, "tags": "#AI슬로프 #SNS #교육"},

    {"date": today - timedelta(days=3), "category": "논문",
     "title": "AI 코딩 어시스턴트 생산성 실험 결과",
     "source": "Stanford HAI", "author_trust": 94, "innovation_score": 67,
     "summary": "스탠퍼드 연구팀이 500명 개발자 대상 6개월 종단 연구를 발표했다.\nAI 코딩 도구 사용 그룹은 태스크 완료 속도 41% 향상을 보였다.\n단, 복잡한 아키텍처 설계에서는 AI 의존이 코드 품질을 저하시킬 수 있다.",
     "url": "https://hai.stanford.edu", "is_bookmarked": False, "tags": "#코딩 #생산성 #연구"},

    {"date": today - timedelta(days=3), "category": "뉴스",
     "title": "뉴로모픽 칩의 부활: 인텔 Loihi 3 공개",
     "source": "IEEE Spectrum", "author_trust": 89, "innovation_score": 97,
     "summary": "인텔이 뇌 구조를 모방한 3세대 뉴로모픽 칩 Loihi 3를 발표했다.\n기존 GPU 대비 에너지 효율이 최대 1000배 높은 특정 AI 태스크를 처리한다.\n스파이킹 신경망(SNN) 상용화의 전환점이 될 것으로 기대된다.",
     "url": "https://spectrum.ieee.org", "is_bookmarked": True, "tags": "#뉴로모픽 #하드웨어 #Intel"},

    {"date": today - timedelta(days=5), "category": "논문",
     "title": "생성 AI와 저작권: 대법원 첫 판결 분석",
     "source": "Harvard Law Review", "author_trust": 98, "innovation_score": 52,
     "summary": "미국 대법원이 AI 생성 콘텐츠의 저작권 귀속에 관한 첫 판결을 내렸다.\n순수 AI 생성물은 저작권 보호 대상이 아니나 인간 창작 기여가 있으면 인정된다.\n이 판결은 라이선스 계약 구조를 전면 재편할 전망이다.",
     "url": "https://harvardlawreview.org", "is_bookmarked": False, "tags": "#저작권 #법률 #규제"},

    {"date": today - timedelta(days=7), "category": "논문",
     "title": "합성 데이터가 실제 데이터를 대체할 수 있을까?",
     "source": "Nature Machine Intelligence", "author_trust": 97, "innovation_score": 83,
     "summary": "고품질 합성 데이터로 훈련한 모델이 실제 데이터 기반 모델과 동등한 성능을 보인다.\n의료·자율주행 등 데이터 희소 도메인에서 잠재력이 특히 크다.\n다만 분포 편향(distribution shift) 문제는 여전히 해결 과제다.",
     "url": "https://www.nature.com/natmachintell", "is_bookmarked": True, "tags": "#합성데이터 #훈련 #연구"},

    {"date": today - timedelta(days=14), "category": "뉴스",
     "title": "AI 에이전트 보안: 프롬프트 인젝션 위협",
     "source": "OWASP Foundation", "author_trust": 90, "innovation_score": 71,
     "summary": "OWASP가 LLM 에이전트 대상 프롬프트 인젝션 공격 10대 패턴을 발표했다.\n외부 데이터 소스를 처리하는 에이전트가 특히 취약하며 피해 사례도 보고됐다.\n입력 샌드박싱·권한 최소화·행동 감사 로그 등이 방어책으로 제시됐다.",
     "url": "https://owasp.org", "is_bookmarked": True, "tags": "#보안 #에이전트 #프롬프트"},
]

mock_data = pd.DataFrame(raw)
mock_data["date"] = pd.to_datetime(mock_data["date"])

# ──────────────────────────────────────────────
# 5. 헬퍼 함수
# ──────────────────────────────────────────────

def build_tfidf_matrix(df: pd.DataFrame):
    """
    TF-IDF 벡터 행렬 생성.

    각 문서(카드)를 title + summary + tags 를 합친 텍스트로 표현한 뒤
    TfidfVectorizer로 단어 빈도-역문서빈도 행렬을 만든다.

    TF-IDF란?
    - TF(Term Frequency): 해당 문서에서 단어가 얼마나 자주 나오는지
    - IDF(Inverse Document Frequency): 전체 문서에서 희귀할수록 높은 가중치
    - 둘을 곱하면 '이 문서에서 중요한 단어'를 수치로 표현할 수 있음
    """
    # 각 카드의 텍스트 컨텍스트 생성 (제목 + 요약 + 태그를 하나의 문자열로)
    corpus = (
        df["title"].fillna("") + " " +
        df["summary"].fillna("").str.replace("\n", " ") + " " +
        df["tags"].fillna("").str.replace("#", " ")
    ).tolist()

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",   # 한국어는 형태소 분석기 없이 음절(char) 단위가 효과적
        ngram_range=(2, 4),   # 2~4글자 n-gram으로 한국어 패턴 포착
        min_df=1,
        max_features=500,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer, corpus


def compute_tfidf_similarity(df: pd.DataFrame, query_text: str) -> np.ndarray:
    """
    쿼리 텍스트와 전체 문서 간 코사인 유사도 계산.

    코사인 유사도란?
    - 두 벡터가 이루는 각도의 코사인 값 (0 ~ 1)
    - 1에 가까울수록 두 문서의 내용이 비슷함
    - 방향(내용의 유사성)만 비교하므로 문서 길이 차이에 영향 받지 않음

    슬라이더가 없을 때(쿼리 없음): 전체 문서 간 평균 유사도로 중심성 계산
    슬라이더가 있을 때: 슬라이더 성향 텍스트와의 유사도 계산
    """
    tfidf_matrix, vectorizer, _ = build_tfidf_matrix(df)

    if query_text.strip():
        # 쿼리 텍스트를 같은 벡터 공간으로 변환
        query_vec = vectorizer.transform([query_text])
        # 쿼리와 각 문서 간 코사인 유사도 (1 × N 배열)
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    else:
        # 쿼리 없음 → 문서 간 평균 코사인 유사도 (중심성: 전체와 얼마나 관련 있는지)
        sim_matrix = cosine_similarity(tfidf_matrix)
        sim_scores = sim_matrix.mean(axis=1)

    return sim_scores


def compute_total_score(df: pd.DataFrame, slider_val: int, query: str = "") -> pd.DataFrame:
    """
    TF-IDF 코사인 유사도 + 슬라이더 가중치를 결합한 최종 점수 계산.

    ① TF-IDF 유사도 점수 (0~1) → 0~100으로 정규화
       - 쿼리 있음: 검색어와 내용이 얼마나 비슷한지
       - 쿼리 없음: 전체 코퍼스에서 얼마나 대표적인 내용인지 (중심성)

    ② 슬라이더 가중치로 권위/혁신 점수 혼합
       - slider=0   → 권위 100%
       - slider=100 → 혁신 100%
       - 중간값은 비율로 혼합

    ③ 최종 점수 = TF-IDF 유사도(40%) + 슬라이더 혼합 점수(60%)
       → TF-IDF가 내용 기반 관련성, 슬라이더가 성향 기반 선호도를 반영
    """
    df = df.copy()

    # ── ① TF-IDF 코사인 유사도 계산 ──
    # 슬라이더 성향을 텍스트 쿼리로 변환해 유사도 방향 결정
    if query.strip():
        # 사용자가 직접 검색어 입력한 경우
        sim_text = query
    elif slider_val >= 70:
        # 혁신 성향: 혁신·신기술 관련 키워드로 유사도 방향 설정
        sim_text = "혁신 신기술 새로운 발견 미래 실험 창발 혁신적 엣지 신선"
    elif slider_val <= 30:
        # 권위 성향: 신뢰·권위 관련 키워드로 유사도 방향 설정
        sim_text = "연구 분석 보고서 공식 학술 데이터 검증 신뢰 권위 안전"
    else:
        # 중간 성향: 쿼리 없이 중심성 기반 계산
        sim_text = ""

    sim_scores = compute_tfidf_similarity(df, sim_text)

    # 유사도 점수 0~100 정규화
    sim_min, sim_max = sim_scores.min(), sim_scores.max()
    if sim_max > sim_min:
        sim_normalized = (sim_scores - sim_min) / (sim_max - sim_min) * 100
    else:
        sim_normalized = np.ones(len(df)) * 50  # 모두 동일하면 50점

    df["tfidf_score"] = sim_normalized.round(1)

    # ── ② 슬라이더 가중치 혼합 점수 ──
    w_i = slider_val / 100          # 혁신 가중치
    w_t = 1 - w_i                   # 권위 가중치
    df["weighted_score"] = (df["author_trust"] * w_t + df["innovation_score"] * w_i).round(1)

    # ── ③ 최종 점수: TF-IDF 40% + 가중 혼합 60% ──
    df["total_score"] = (
        df["tfidf_score"]    * 0.4 +
        df["weighted_score"] * 0.6
    ).round(1)

    return df.sort_values("total_score", ascending=False)


def gauge_html(trust: int, innov: int, total: float, tfidf: float = 0.0) -> str:
    """권위·혁신·TF-IDF·종합 점수를 게이지 바 HTML로 변환."""
    def bar(label, val, color):
        return (
            f'<div class="gauge-row">'
            f'<span class="gauge-label">{label}</span>'
            f'<div class="gauge-bg"><div class="gauge-fill" style="width:{val}%;background:{color};"></div></div>'
            f'<span class="gauge-val" style="color:{color};">{val}</span>'
            f'</div>'
        )
    return (
        '<div class="gauge-wrap">'
        + bar("🔒 권위", trust, "#4ecca3")
        + bar("🚀 혁신", innov, "#f5627a")
        + bar("🧮 유사도", int(tfidf), "#f5a623")
        + bar("⭐ 종합", min(int(total), 100), "#5b8af5")
        + '</div>'
    )


def cat_badge_html(category: str) -> str:
    mapping = {"뉴스": ("cat-news","뉴스"), "논문": ("cat-paper","논문"), "SNS/트렌드": ("cat-sns","SNS·트렌드")}
    cls, label = mapping.get(category, ("cat-news", category))
    return f'<span class="cat-badge {cls}">{label}</span>'


def render_card(row, idx, card_class="", show_source_box=False, tab_key=""):
    """카드 렌더링 + 읽음 토글 버튼. tab_key로 버튼 key 충돌 방지."""
    is_read    = st.session_state.read_status.get(idx, False)
    read_extra = "is-read" if is_read else ""
    read_label = "✓ 읽음" if is_read else "● 안읽음"
    badge_cls  = "read" if is_read else "unread"
    tags_html  = "".join(f'<span class="tag">{t.strip()}</span>' for t in str(row["tags"]).split("#") if t.strip())
    summary    = "<br>".join(str(row["summary"]).split("\n"))
    tfidf_val  = float(row.get("tfidf_score", 0))
    gauges     = gauge_html(int(row["author_trust"]), int(row["innovation_score"]), float(row["total_score"]), tfidf_val)
    cat_b      = cat_badge_html(str(row.get("category", "")))
    date_str   = row["date"].strftime("%Y-%m-%d")

    if show_source_box:
        link_area = (
            f'<div style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;'
            f'padding:0.55rem 0.9rem;display:flex;justify-content:space-between;align-items:center;'
            f'flex-wrap:wrap;gap:0.4rem;margin-top:0.5rem;">'
            f'<span style="font-size:0.8rem;">📰 출처 &nbsp;<strong style="color:var(--trust);">{row["source"]}</strong></span>'
            f'<a href="{row["url"]}" target="_blank" style="font-size:0.78rem;color:var(--accent);text-decoration:none;'
            f'background:rgba(91,138,245,0.12);border:1px solid rgba(91,138,245,0.3);border-radius:6px;'
            f'padding:0.22rem 0.65rem;font-weight:500;">🔗 원문 바로가기 →</a></div>'
        )
    else:
        link_area = (f'<div style="margin-top:0.4rem;">'
                     f'<a href="{row["url"]}" target="_blank" style="font-size:0.78rem;color:var(--accent);text-decoration:none;">'
                     f'🔗 원문 보기 →</a></div>')

    st.markdown(f"""
    <div class="card {card_class} {read_extra}">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.5rem;margin-bottom:0.4rem;">
            <div class="card-title">{row['title']}</div>
            <span class="read-badge {badge_cls}" style="flex-shrink:0;">{read_label}</span>
        </div>
        <div class="card-meta">{cat_b}<span>📰 {row['source']}</span><span>📅 {date_str}</span></div>
        <div class="card-summary">{summary}</div>
        {gauges}
        <div style="margin-bottom:0.4rem;">{tags_html}</div>
        {link_area}
    </div>
    """, unsafe_allow_html=True)

    btn_label = "📖 안읽음으로 표시" if is_read else "✅ 읽음으로 표시"
    col_btn, col_empty = st.columns([1, 4])
    with col_btn:
        if st.button(btn_label, key=f"read_btn_{tab_key}_{idx}", use_container_width=True):
            st.session_state.read_status[idx] = not is_read
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()


def read_summary_bar(data: pd.DataFrame, prefix: str = ""):
    total  = len(data)
    read_n = sum(st.session_state.read_status.get(i, False) for i in data.index)
    st.markdown(
        f'<div style="display:flex;gap:0.7rem;margin-bottom:1rem;flex-wrap:wrap;align-items:center;">'
        f'<span style="font-size:0.8rem;color:var(--muted);">{prefix}총 {total}건</span>'
        f'<span class="read-badge unread">● 안읽음 {total - read_n}</span>'
        f'<span class="read-badge read">✓ 읽음 {read_n}</span></div>',
        unsafe_allow_html=True,
    )


def hbar_chart(items: list, color: str, title: str):
    """수평 바 차트 HTML 렌더링. items = [(label, value), ...]"""
    if not items:
        return
    max_val = max(v for _, v in items) or 1
    bars = ""
    for label, val in items:
        pct = int(val / max_val * 100)
        bars += (
            f'<div class="hbar-row">'
            f'<span class="hbar-label">{label}</span>'
            f'<div class="hbar-bg"><div class="hbar-fill" style="width:{pct}%;background:{color};"></div></div>'
            f'<span class="hbar-val" style="color:{color};">{val}</span>'
            f'</div>'
        )
    st.markdown(
        f'<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);'
        f'padding:1rem 1.2rem;margin-bottom:0.75rem;">'
        f'<div class="section-label" style="margin-bottom:0.65rem;">{title}</div>'
        f'{bars}</div>',
        unsafe_allow_html=True,
    )


def stat_card(value, label, sub="", color="#5b8af5"):
    """숫자 요약 카드 HTML."""
    return (
        f'<div class="stat-card">'
        f'<div class="stat-value" style="color:{color};">{value}</div>'
        f'<div class="stat-label">{label}</div>'
        f'{"<div class=stat-sub>" + sub + "</div>" if sub else ""}'
        f'</div>'
    )


# ──────────────────────────────────────────────
# 6. 앱 헤더
# ──────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">🧠 AI 정보 비서</div>
    <div class="app-subtitle">PERSONAL AI INFORMATION CURATOR · 맞춤형 정보 탐색 시스템 v3</div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 7. 상단 컨트롤 패널
# ──────────────────────────────────────────────
st.markdown('<div class="control-panel">', unsafe_allow_html=True)
col_search, col_slider = st.columns([1.4, 1], gap="large")

with col_search:
    st.markdown('<div class="section-label">정보 입력</div>', unsafe_allow_html=True)
    keyword = st.text_input(
        label="검색창",
        placeholder="🔍  URL 입력 또는 키워드 검색 (예: LLM, 투자, 규제 …)",
        label_visibility="collapsed",
    )
    if keyword:
        st.caption(f"ℹ️ '{keyword}' — 다음 버전에서 실제 AI 분석과 연동됩니다.")

with col_slider:
    st.markdown('<div class="section-label">정보 탐색 성향</div>', unsafe_allow_html=True)
    slider_val = st.slider(
        label="성향", min_value=0, max_value=100, value=50, step=5,
        label_visibility="collapsed",
        help="0 = 권위·신뢰 위주  /  100 = 신선함·혁신 위주",
    )
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;font-size:0.74rem;margin-top:4px;">'
        f'<span style="color:#4ecca3;">🔒 권위 {100 - slider_val}%</span>'
        f'<span style="color:#f5627a;">🚀 혁신 {slider_val}%</span></div>',
        unsafe_allow_html=True,
    )
st.markdown('</div>', unsafe_allow_html=True)

# 슬라이더 + 검색 키워드를 모두 반영해 TF-IDF 기반 점수 계산
scored_data = compute_total_score(mock_data, slider_val, query=keyword)

# ──────────────────────────────────────────────
# 8. 상위 탭 (정보 3개 + 통계 1개)
# ──────────────────────────────────────────────
tab_today, tab_past, tab_bookmark, tab_stats = st.tabs([
    "📌  오늘의 정보",
    "🗂  이전 정보",
    "⭐  즐겨찾기",
    "📊  통계 대시보드",
])

# ════════════════════════════════════════
# 탭 1 — 오늘의 정보
# ════════════════════════════════════════
with tab_today:
    today_data = scored_data[scored_data["date"].dt.date == today].copy()
    if today_data.empty:
        st.markdown('<div class="empty-state">📭 오늘 날짜의 정보가 없습니다.</div>', unsafe_allow_html=True)
    else:
        read_summary_bar(today_data, "오늘 · ")
        half          = max(1, len(today_data) // 2)
        core_idx      = set(today_data.sort_values("author_trust",     ascending=False).head(half).index)
        discovery_idx = set(today_data.sort_values("innovation_score", ascending=False).head(half).index)

        sub_tabs = st.tabs(["🗃 전체", "📰 뉴스", "📄 논문", "📢 SNS·트렌드"])
        for sub_tab, cat in zip(sub_tabs, [None, "뉴스", "논문", "SNS/트렌드"]):
            with sub_tab:
                cat_str  = cat if cat else "전체"
                tab_key  = f"today_{cat_str}"
                filtered = today_data if cat is None else today_data[today_data["category"] == cat]
                if filtered.empty:
                    st.markdown('<div class="empty-state">해당 카테고리의 정보가 없습니다.</div>', unsafe_allow_html=True)
                    continue
                core_f      = filtered[filtered.index.isin(core_idx)]
                discovery_f = filtered[filtered.index.isin(discovery_idx)]
                if not core_f.empty:
                    st.markdown('<div class="section-label trust">🔒 핵심 정보 — 권위·신뢰 기반</div>', unsafe_allow_html=True)
                    for idx, row in core_f.iterrows():
                        render_card(row, idx, "trust-card", tab_key=f"{tab_key}_core")
                if not core_f.empty and not discovery_f.empty:
                    st.markdown('<hr class="divider">', unsafe_allow_html=True)
                if not discovery_f.empty:
                    st.markdown('<div class="section-label innov">🚀 오늘의 발견 — 혁신·신선함</div>', unsafe_allow_html=True)
                    for idx, row in discovery_f.iterrows():
                        render_card(row, idx, "innov-card", tab_key=f"{tab_key}_disc")

        with st.expander("📊 오늘 데이터 전체 테이블"):
            disp = today_data[["title","category","source","author_trust","innovation_score","total_score","tags"]].reset_index(drop=True)
            st.dataframe(disp, use_container_width=True, hide_index=True,
                         column_config={
                             "author_trust":     st.column_config.ProgressColumn("권위", min_value=0, max_value=100),
                             "innovation_score": st.column_config.ProgressColumn("혁신", min_value=0, max_value=100),
                             "total_score":      st.column_config.NumberColumn("종합", format="%.1f"),
                         })

# ════════════════════════════════════════
# 탭 2 — 이전 정보
# ════════════════════════════════════════
with tab_past:
    past_dates = sorted(
        scored_data[scored_data["date"].dt.date < today]["date"].dt.date.unique(), reverse=True)
    col_cal, col_hint = st.columns([1, 2], gap="medium")
    with col_cal:
        selected_date = st.date_input(
            label="날짜 선택",
            value=past_dates[0] if past_dates else today - timedelta(days=1),
            min_value=past_dates[-1] if past_dates else today - timedelta(days=30),
            max_value=today - timedelta(days=1),
            label_visibility="collapsed",
        )
    with col_hint:
        if past_dates:
            st.caption(f"📅 데이터 보유: {', '.join(str(d) for d in past_dates)}")
    st.markdown("<br>", unsafe_allow_html=True)
    past_data = scored_data[scored_data["date"].dt.date == selected_date].copy()
    if past_data.empty:
        st.markdown(f'<div class="empty-state">📭 {selected_date} 날짜의 정보가 없습니다.</div>', unsafe_allow_html=True)
    else:
        read_summary_bar(past_data, f"{selected_date} · ")
        sub_tabs2 = st.tabs(["🗃 전체", "📰 뉴스", "📄 논문", "📢 SNS·트렌드"])
        for sub_tab, cat in zip(sub_tabs2, [None, "뉴스", "논문", "SNS/트렌드"]):
            with sub_tab:
                cat_str  = cat if cat else "전체"
                tab_key  = f"past_{selected_date}_{cat_str}"
                filtered = past_data if cat is None else past_data[past_data["category"] == cat]
                if filtered.empty:
                    st.markdown('<div class="empty-state">해당 카테고리의 정보가 없습니다.</div>', unsafe_allow_html=True)
                    continue
                st.markdown(f'<div class="section-label">{len(filtered)}건 — 출처 및 원문 포함</div>', unsafe_allow_html=True)
                for idx, row in filtered.iterrows():
                    render_card(row, idx, "trust-card", show_source_box=True, tab_key=tab_key)

# ════════════════════════════════════════
# 탭 3 — 즐겨찾기
# ════════════════════════════════════════
with tab_bookmark:
    bookmarked = scored_data[scored_data["is_bookmarked"] == True].copy()
    if bookmarked.empty:
        st.markdown('<div class="empty-state">즐겨찾기된 정보가 없습니다.</div>', unsafe_allow_html=True)
    else:
        all_tags: list = []
        for tags_str in bookmarked["tags"]:
            for t in str(tags_str).split("#"):
                t = t.strip()
                if t and t not in all_tags:
                    all_tags.append(t)

        col_f, col_c = st.columns([2, 1], gap="medium")
        with col_f:
            selected_tags = st.multiselect(
                label="태그 선택 (복수 가능) — 비워두면 전체 표시",
                options=sorted(all_tags), default=[],
            )
        with col_c:
            bkmk_filtered = (
                bookmarked[bookmarked["tags"].apply(lambda t: any(tag in t for tag in selected_tags))]
                if selected_tags else bookmarked
            )
            read_b   = sum(st.session_state.read_status.get(i, False) for i in bkmk_filtered.index)
            unread_b = len(bkmk_filtered) - read_b
            st.markdown(
                f'<div style="font-size:1.25rem;font-weight:700;color:var(--accent2);">'
                f'{len(bkmk_filtered)}'
                f'<span style="font-size:0.76rem;color:var(--muted);font-weight:400;"> 건 &nbsp;</span>'
                f'<span class="read-badge unread" style="font-size:0.65rem;">● {unread_b}</span> '
                f'<span class="read-badge read" style="font-size:0.65rem;">✓ {read_b}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)
        if bkmk_filtered.empty:
            st.markdown('<div class="empty-state">선택한 태그에 해당하는 즐겨찾기가 없습니다.</div>', unsafe_allow_html=True)
        else:
            sub_tabs3 = st.tabs(["🗃 전체", "📰 뉴스", "📄 논문", "📢 SNS·트렌드"])
            for sub_tab, cat in zip(sub_tabs3, [None, "뉴스", "논문", "SNS/트렌드"]):
                with sub_tab:
                    cat_str  = cat if cat else "전체"
                    tab_key  = f"bkmk_{cat_str}"
                    filtered = bkmk_filtered if cat is None else bkmk_filtered[bkmk_filtered["category"] == cat]
                    if filtered.empty:
                        st.markdown('<div class="empty-state">해당 카테고리의 즐겨찾기가 없습니다.</div>', unsafe_allow_html=True)
                        continue
                    for idx, row in filtered.iterrows():
                        render_card(row, idx, "bkmk-card", tab_key=tab_key)

        with st.expander("📊 즐겨찾기 전체 테이블"):
            disp2 = bkmk_filtered[["date","title","category","source","author_trust","innovation_score","total_score","tags"]].reset_index(drop=True)
            st.dataframe(disp2, use_container_width=True, hide_index=True,
                         column_config={
                             "date":             st.column_config.DateColumn("날짜", format="YYYY-MM-DD"),
                             "author_trust":     st.column_config.ProgressColumn("권위", min_value=0, max_value=100),
                             "innovation_score": st.column_config.ProgressColumn("혁신", min_value=0, max_value=100),
                             "total_score":      st.column_config.NumberColumn("종합", format="%.1f"),
                         })

# ════════════════════════════════════════
# 탭 4 — 통계 대시보드
# ════════════════════════════════════════
with tab_stats:

    # ── 전처리: 태그 카운트 ──
    tag_counter: dict = {}
    for tags_str in scored_data["tags"]:
        for t in str(tags_str).split("#"):
            t = t.strip()
            if t:
                tag_counter[t] = tag_counter.get(t, 0) + 1

    # 일별 수집량
    daily_counts = (
        scored_data.groupby(scored_data["date"].dt.date)
        .size().reset_index(name="count")
        .sort_values("date")
    )

    # 일별 평균 점수
    daily_scores = (
        scored_data.groupby(scored_data["date"].dt.date)[["author_trust","innovation_score"]]
        .mean().round(1).reset_index().sort_values("date")
    )

    # 출처별 평균 권위 점수
    source_trust = (
        scored_data.groupby("source")["author_trust"]
        .mean().round(1).sort_values(ascending=False).head(8)
    )

    # 카테고리 분포
    cat_counts = scored_data["category"].value_counts()

    # 전체 읽음 수
    total_cnt  = len(scored_data)
    read_cnt   = sum(st.session_state.read_status.get(i, False) for i in scored_data.index)
    read_rate  = round(read_cnt / total_cnt * 100) if total_cnt else 0

    # ────────────────────────────────────────
    # 1순위: 핵심 요약 지표 4개
    # ────────────────────────────────────────
    st.markdown('<div class="section-label">핵심 요약 지표</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="small")

    avg_trust  = round(scored_data["author_trust"].mean(), 1)
    avg_innov  = round(scored_data["innovation_score"].mean(), 1)
    daily_avg  = round(daily_counts["count"].mean(), 1)

    with c1:
        st.markdown(stat_card(total_cnt, "전체 수집 건수", f"일 평균 {daily_avg}건", "#5b8af5"), unsafe_allow_html=True)
    with c2:
        st.markdown(stat_card(f"{read_rate}%", "전체 읽음률", f"{read_cnt}/{total_cnt}건 읽음", "#4ecca3"), unsafe_allow_html=True)
    with c3:
        st.markdown(stat_card(avg_trust, "평균 권위 점수", f"최고 {int(scored_data['author_trust'].max())} / 최저 {int(scored_data['author_trust'].min())}", "#4ecca3"), unsafe_allow_html=True)
    with c4:
        st.markdown(stat_card(avg_innov, "평균 혁신 점수", f"최고 {int(scored_data['innovation_score'].max())} / 최저 {int(scored_data['innovation_score'].min())}", "#f5627a"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ────────────────────────────────────────
    # 2순위 상단: 태그 빈도 바 차트 + 출처별 신뢰도
    # ────────────────────────────────────────
    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        # 태그 빈도 Top 10 수평 바 차트
        top_tags = sorted(tag_counter.items(), key=lambda x: x[1], reverse=True)[:10]
        hbar_chart(top_tags, "#5b8af5", "🏷 태그 빈도 Top 10")

    with col_right:
        # 출처별 평균 권위 점수 수평 바 차트
        source_items = [(src, val) for src, val in source_trust.items()]
        hbar_chart(source_items, "#4ecca3", "📰 출처별 평균 권위 점수 Top 8")

    st.markdown("<br>", unsafe_allow_html=True)

    # ────────────────────────────────────────
    # 2순위 중단: 일별 수집량 + 권위/혁신 점수 추이 (st.line_chart)
    # ────────────────────────────────────────
    col_a, col_b = st.columns(2, gap="medium")

    with col_a:
        st.markdown(
            '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);'
            'padding:1rem 1.2rem;margin-bottom:0.75rem;">'
            '<div class="section-label" style="margin-bottom:0.5rem;">📅 일별 정보 수집량</div>',
            unsafe_allow_html=True,
        )
        chart_df = daily_counts.set_index("date").rename(columns={"count": "수집 건수"})
        st.line_chart(chart_df, height=200, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown(
            '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);'
            'padding:1rem 1.2rem;margin-bottom:0.75rem;">'
            '<div class="section-label" style="margin-bottom:0.5rem;">📈 일별 평균 점수 추이</div>',
            unsafe_allow_html=True,
        )
        score_chart = daily_scores.set_index("date").rename(
            columns={"author_trust": "권위 점수", "innovation_score": "혁신 점수"})
        st.line_chart(score_chart, height=200, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ────────────────────────────────────────
    # 2순위 하단: 카테고리 분포 + 권위 vs 혁신 산점도
    # ────────────────────────────────────────
    col_c, col_d = st.columns(2, gap="medium")

    with col_c:
        # 카테고리 분포 바 차트
        cat_items = [(cat, cnt) for cat, cnt in cat_counts.items()]
        cat_colors = {"뉴스": "#5b8af5", "논문": "#4ecca3", "SNS/트렌드": "#f5a623"}
        st.markdown(
            '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);'
            'padding:1rem 1.2rem;">'
            '<div class="section-label" style="margin-bottom:0.65rem;">🗂 카테고리별 수집 건수</div>',
            unsafe_allow_html=True,
        )
        max_cat = max(cnt for _, cnt in cat_items) or 1
        for cat, cnt in cat_items:
            pct = int(cnt / max_cat * 100)
            col_hex = cat_colors.get(cat, "#5b8af5").replace("#","")
            st.markdown(
                f'<div class="hbar-row">'
                f'<span class="hbar-label">{cat}</span>'
                f'<div class="hbar-bg"><div class="hbar-fill" style="width:{pct}%;background:#{col_hex};"></div></div>'
                f'<span class="hbar-val" style="color:#{col_hex};">{cnt}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_d:
        # 권위 vs 혁신 산점도 — HTML/SVG로 직접 구현 (구버전 Streamlit 호환)
        corr = scored_data["author_trust"].corr(scored_data["innovation_score"])
        corr_label = '(약한 관계)' if abs(corr) < 0.3 else '(중간 관계)' if abs(corr) < 0.6 else '(강한 관계)'

        cat_color_map = {"뉴스": "#5b8af5", "논문": "#4ecca3", "SNS/트렌드": "#f5a623"}
        W, H, PAD = 320, 220, 28  # SVG 캔버스 크기 및 패딩

        dots = ""
        for _, r in scored_data.iterrows():
            # 점수를 SVG 좌표로 변환 (x=권위, y=혁신, y축은 위가 높은 값)
            cx = PAD + (r["author_trust"] - 40) / 60 * (W - PAD*2)
            cy = H - PAD - (r["innovation_score"] - 40) / 60 * (H - PAD*2)
            col_hex = cat_color_map.get(r["category"], "#5b8af5")
            title_safe = str(r["title"])[:18].replace("<","&lt;").replace(">","&gt;")
            dots += (
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" '
                f'fill="{col_hex}" fill-opacity="0.85" stroke="#0d0f14" stroke-width="1.2">'
                f'<title>{title_safe}</title></circle>'
            )

        # 축 눈금선
        grid = ""
        for v in [40, 55, 70, 85, 100]:
            xg = PAD + (v - 40) / 60 * (W - PAD*2)
            yg = H - PAD - (v - 40) / 60 * (H - PAD*2)
            yg_label = yg + 4   # f-string 안에서 연산 불가하므로 미리 계산
            grid += (f'<line x1="{xg:.1f}" y1="{PAD}" x2="{xg:.1f}" y2="{H-PAD}" '
                     f'stroke="#2a303f" stroke-width="1"/>')
            grid += (f'<line x1="{PAD}" y1="{yg:.1f}" x2="{W-PAD}" y2="{yg:.1f}" '
                     f'stroke="#2a303f" stroke-width="1"/>')
            grid += (f'<text x="{xg:.1f}" y="{H-PAD+12}" text-anchor="middle" '
                     f'font-size="8" fill="#7a8099">{v}</text>')
            grid += (f'<text x="{PAD-4}" y="{yg_label:.1f}" text-anchor="end" '
                     f'font-size="8" fill="#7a8099">{v}</text>')

        # 범례
        legend = ""
        for i, (cat, col) in enumerate(cat_color_map.items()):
            legend += (f'<circle cx="{PAD + i*90}" cy="12" r="5" fill="{col}"/>'
                       f'<text x="{PAD + i*90 + 9}" y="16" font-size="9" fill="#b0b5c8">{cat}</text>')

        svg = f"""
        <svg width="{W}" height="{H+20}" xmlns="http://www.w3.org/2000/svg"
             style="background:#161923;border-radius:8px;width:100%;">
          {grid}{dots}{legend}
          <text x="{W//2}" y="{H+14}" text-anchor="middle" font-size="9" fill="#7a8099">권위 점수 →</text>
          <text x="10" y="{H//2}" text-anchor="middle" font-size="9" fill="#7a8099"
                transform="rotate(-90,10,{H//2})">혁신 점수 →</text>
        </svg>
        """

        st.markdown(
            f'<div style="background:var(--surface);border:1px solid var(--border);'
            f'border-radius:var(--radius);padding:1rem 1.2rem;">'
            f'<div class="section-label" style="margin-bottom:0.5rem;">🔬 권위 vs 혁신 점수 산점도</div>'
            f'{svg}'
            f'<div style="font-size:0.75rem;color:var(--muted);margin-top:0.4rem;">'
            f'Pearson r = {corr:.3f} {corr_label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ────────────────────────────────────────
    # 보너스: 정보 다양성 지수 (Shannon Entropy)
    # ────────────────────────────────────────
    st.markdown('<div class="section-label">고급 지표</div>', unsafe_allow_html=True)
    col_e, col_f2, col_g = st.columns(3, gap="small")

    # Shannon Entropy — 태그 다양성
    tag_total = sum(tag_counter.values())
    entropy = 0.0
    for cnt in tag_counter.values():
        p = cnt / tag_total
        if p > 0:
            entropy -= p * math.log2(p)
    max_entropy = math.log2(len(tag_counter)) if len(tag_counter) > 1 else 1
    diversity_pct = round(entropy / max_entropy * 100, 1) if max_entropy > 0 else 0

    # 이상치 탐지 (평균 ± 2σ 벗어난 건수)
    trust_mean = scored_data["author_trust"].mean()
    trust_std  = scored_data["author_trust"].std()
    innov_mean = scored_data["innovation_score"].mean()
    innov_std  = scored_data["innovation_score"].std()
    outliers   = scored_data[
        (scored_data["author_trust"]     > trust_mean + 2*trust_std) |
        (scored_data["author_trust"]     < trust_mean - 2*trust_std) |
        (scored_data["innovation_score"] > innov_mean + 2*innov_std) |
        (scored_data["innovation_score"] < innov_mean - 2*innov_std)
    ]

    # 즐겨찾기율
    bkmk_rate = round(scored_data["is_bookmarked"].mean() * 100, 1)

    with col_e:
        st.markdown(
            stat_card(f"{diversity_pct}%", "정보 다양성 지수",
                      f"Shannon Entropy {entropy:.2f} bits", "#f5a623"),
            unsafe_allow_html=True,
        )
    with col_f2:
        st.markdown(
            stat_card(len(outliers), "이상치 건수",
                      f"평균 ± 2σ 기준 | 전체의 {round(len(outliers)/total_cnt*100)}%", "#f5627a"),
            unsafe_allow_html=True,
        )
    with col_g:
        st.markdown(
            stat_card(f"{bkmk_rate}%", "즐겨찾기율",
                      f"전체 {total_cnt}건 중 {scored_data['is_bookmarked'].sum()}건 북마크", "#4ecca3"),
            unsafe_allow_html=True,
        )

    # ────────────────────────────────────────
    # TF-IDF 알고리즘 설명 섹션
    # ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">🧮 TF-IDF 알고리즘 작동 원리</div>', unsafe_allow_html=True)

    col_algo1, col_algo2 = st.columns(2, gap="medium")
    with col_algo1:
        st.markdown("""
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.1rem 1.3rem;">
        <div style="font-size:0.78rem;font-weight:700;color:var(--accent);margin-bottom:0.6rem;">📐 점수 계산 공식</div>
        <div style="font-size:0.82rem;color:var(--muted);line-height:1.8;">
        <b style="color:var(--text);">① TF-IDF 유사도 (40%)</b><br>
        &nbsp;· 제목 + 요약 + 태그를 텍스트 벡터로 변환<br>
        &nbsp;· 슬라이더 성향 키워드와 코사인 유사도 계산<br>
        &nbsp;· 0~100으로 정규화 → <b style="color:var(--accent2);">🧮 유사도 점수</b><br><br>
        <b style="color:var(--text);">② 슬라이더 가중 혼합 (60%)</b><br>
        &nbsp;· 권위 × (1 - slider/100)<br>
        &nbsp;· 혁신 × (slider/100)<br><br>
        <b style="color:var(--text);">③ 최종 점수</b><br>
        &nbsp;· TF-IDF × 0.4 + 가중혼합 × 0.6
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col_algo2:
        # 현재 슬라이더 값에 따라 어떤 키워드로 유사도를 계산하는지 표시
        if keyword.strip():
            sim_direction = f'사용자 검색어: "{keyword}"'
            sim_color = "#5b8af5"
        elif slider_val >= 70:
            sim_direction = "혁신 성향 키워드 (혁신·신기술·창발·미래…)"
            sim_color = "#f5627a"
        elif slider_val <= 30:
            sim_direction = "권위 성향 키워드 (연구·학술·검증·신뢰…)"
            sim_color = "#4ecca3"
        else:
            sim_direction = "중립 — 전체 문서 중심성 기반"
            sim_color = "#f5a623"

        # TF-IDF 점수 상위 5개 표시
        top5 = scored_data.nlargest(5, "tfidf_score")[["title", "tfidf_score", "total_score"]]

        st.markdown(f"""
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.1rem 1.3rem;">
        <div style="font-size:0.78rem;font-weight:700;color:var(--accent);margin-bottom:0.6rem;">🎯 현재 유사도 방향</div>
        <div style="font-size:0.82rem;color:{sim_color};font-weight:600;margin-bottom:0.8rem;">
            {sim_direction}
        </div>
        <div style="font-size:0.75rem;font-weight:700;color:var(--muted);margin-bottom:0.4rem;">
            TF-IDF 유사도 Top 5
        </div>
        """, unsafe_allow_html=True)

        for _, r in top5.iterrows():
            pct = int(r["tfidf_score"])
            title_short = r["title"][:20] + "…" if len(r["title"]) > 20 else r["title"]
            st.markdown(
                f'<div class="hbar-row">'
                f'<span class="hbar-label" style="width:130px;font-size:0.72rem;">{title_short}</span>'
                f'<div class="hbar-bg"><div class="hbar-fill" style="width:{pct}%;background:{sim_color};"></div></div>'
                f'<span class="hbar-val" style="color:{sim_color};">{pct}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)
        with st.expander(f"🔍 이상치 카드 목록 ({len(outliers)}건) — 평균 ± 2σ 벗어난 항목"):
            disp_out = outliers[["title","category","source","author_trust","innovation_score","total_score"]].reset_index(drop=True)
            st.dataframe(disp_out, use_container_width=True, hide_index=True,
                         column_config={
                             "author_trust":     st.column_config.ProgressColumn("권위", min_value=0, max_value=100),
                             "innovation_score": st.column_config.ProgressColumn("혁신", min_value=0, max_value=100),
                             "total_score":      st.column_config.NumberColumn("종합", format="%.1f"),
                         })

# ──────────────────────────────────────────────
# 9. 푸터
# ──────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;
            border-top:1px solid var(--border);margin-top:2rem;
            font-size:0.75rem;color:var(--muted);">
    🧠 AI 정보 비서 v3 &nbsp;·&nbsp; Powered by Streamlit &nbsp;·&nbsp;
    슬라이더로 탐색 성향 조절 &nbsp;·&nbsp; 통계 탭에서 인사이트 확인
</div>
""", unsafe_allow_html=True)
