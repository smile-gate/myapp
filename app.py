import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, json, time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from supabase import create_client, Client

# ─────────────────────────────────────────────
# 0. 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="겸업 심사 AI 예측 시스템",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {font-size:2rem; font-weight:700; color:#1a3a5c; margin-bottom:0.2rem;}
    .sub-title {font-size:1rem; color:#555; margin-bottom:1.5rem;}
    .section-header {
        background:#1a3a5c; color:white; padding:0.4rem 0.8rem;
        border-radius:6px; font-weight:600; margin:1rem 0 0.5rem 0;
    }
    .info-bar {
        background:#f8f9fa; border-radius:8px; padding:0.5rem 1rem;
        margin-bottom:0.8rem; font-size:0.82rem; display:flex;
        gap:1.5rem; flex-wrap:wrap; color:#333;
    }
    .info-bar b {color:#1a3a5c; margin-right:0.2rem;}
    .allow-box {
        background:#e6f4ea; border-left:5px solid #34a853;
        padding:0.8rem 1rem; border-radius:6px; font-size:1rem;
        font-weight:600; color:#1e7e34; margin:0.5rem 0;
    }
    .deny-box {
        background:#fce8e6; border-left:5px solid #ea4335;
        padding:0.8rem 1rem; border-radius:6px; font-size:1rem;
        font-weight:600; color:#c0392b; margin:0.5rem 0;
    }
    .caution-box {
        background:#fff8e1; border-left:5px solid #fbbc04;
        padding:0.8rem 1rem; border-radius:6px; font-size:1rem;
        font-weight:600; color:#856404; margin:0.5rem 0;
    }
    .stRadio > div {flex-direction:row; gap:1rem;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 1. 진단항목 정의
# ─────────────────────────────────────────────
ITEMS = [
    ("업무형태(타업종사)",   "진단항목1",  "본인의 사업체를 직접 운영하거나 실질적으로 영위하는가?"),
    ("업무형태(타업종사)",   "진단항목2",  "타 업체에서 직책이나 직위를 맡고 있거나 직무를 수행하는가?"),
    ("업무형태(타업종사)",   "진단항목3",  "배달, 대리운전, 사무보조 등 취업 시간 외에 하는 부업이나 개별 활동인가?"),
    ("업무형태(외부활동)",   "진단항목4",  "상시,정기적 외부기관(대학,학원 등) 출강인가?"),
    ("업무형태(외부활동)",   "진단항목5",  "일회성 또는 일시적 이벤트성 외부기관 출강인가?"),
    ("업무형태(외부활동)",   "진단항목6",  "유튜브, 개인방송, 블로그, 도서 출판, 작곡 등 창작 및 대외활동인가?"),
    ("업무형태(외부활동)",   "진단항목7",  "외부 봉사활동, 공연, 취미나 특기, 과외, 멘토링 등 관련 개인 활동인가?"),
    ("업무형태(외부활동)",   "진단항목8",  "CCL을 통한 게임개발 및 판매 활동인가?"),
    ("업무형태(외부활동)",   "진단항목9",  "CCL을 통한 게임개발 관련 부속 활동인가? (유튜브, 대외행사참가 등)"),
    ("업무시간 및 지장여부", "진단항목10", "업무시간 외 시간을 활용하는가? (=물리적 시간을 의미, 평일 9~18시)"),
    ("업무시간 및 지장여부", "진단항목11", "주당 평균 겸업 투입 시간이 12시간을 초과하는가?"),
    ("업무시간 및 지장여부", "진단항목12", "겸업 수행에 야간/심야 업무(24시~06시)가 포함되어 있는가?"),
    ("업무시간 및 지장여부", "진단항목13", "겸업 수행 시간이 평일 업무 코어 시간(10~16시)과 겹치는가?"),
    ("업무시간 및 지장여부", "진단항목14", "겸업 수행 시간이 주말인가?"),
    ("업무시간 및 지장여부", "진단항목15", "겸업을 위해 본업의 휴가(연차,반차)를 일회성, 간헐적, 한시적으로 사용하는가?"),
    ("업무시간 및 지장여부", "진단항목16", "겸업을 위해 본업의 휴가(연차,반차)를 정기적으로 사용하는가?"),
    ("업무시간 및 지장여부", "진단항목17", "겸업업무가 육체적으로 과도한 노동을 요하는가?"),
    ("업무시간 및 지장여부", "진단항목18", "겸업으로 인해 직무 능률을 떨어뜨릴 우려가 있는가?"),
    ("이해상충",             "진단항목19", "회사의 이익과 상반되는 이익을 취득할 우려가 있는 행위인가?"),
    ("이해상충",             "진단항목20", "담당업무 또는 직무와 관련성이 있는가?"),
    ("이해상충",             "진단항목21", "담당업무 또는 직무와 관련성이 있는 경우 연관된 업종의 겸업인가?"),
    ("이해상충",             "진단항목22", "담당업무를 통해 습득한 기술, 지식, 유·무형 자산을 활용하는가?"),
    ("이해상충",             "진단항목23", "회사의 영업 비밀이나 원천 기술을 활용하는가?"),
    ("이해상충",             "진단항목24", "동종 업계, 산업 혹은 경쟁 관계가 될 수 있거나 경쟁관계인 기업의 업무인가?"),
    ("이해상충",             "진단항목25", "겸업을 통해 얻는 정보나 결과물이 회사에 손실을 끼칠 우려가 있는가?"),
    ("이해상충",             "진단항목26", "현재 회사의 주요 고객사/협력사와 직접적인 계약 관계인가?"),
    ("자산활용",             "진단항목27", "회사의 유무형 자산,장비,장소,시설을 활용하는가? (PPT템플릿, 노트북 등)"),
    ("자산활용",             "진단항목28", "회사의 인력이나 네트워크를 겸업에 동원하는가?"),
    ("평판/브랜드 리스크",   "진단항목29", "해당 업무가 사회 통념상 회사의 명예를 실추시킬 우려가 있는가?"),
    ("평판/브랜드 리스크",   "진단항목30", "직무 수행중에 취득한 비밀을 누설하거나 누설할 우려가 있는가?"),
    ("평판/브랜드 리스크",   "진단항목31", "본인의 소속 회사 및 부서를 공개하는가?"),
    ("평판/브랜드 리스크",   "진단항목32", "회사의 로고, 이름, 사진 등을 직접 활용하는가?"),
    ("수익성",               "진단항목33", "정기적 소득인가?"),
    ("수익성",               "진단항목34", "일회성 소득인가? (상금 등 포함)"),
    ("수익성",               "진단항목35", "본인 사업자 등록이 필요한 형태인가?"),
    ("수익성",               "진단항목36", "타인(가족 등) 명의를 빌려 실질적인 운영권을 행사하는 형태인가?"),
    ("수익성",               "진단항목37", "이중취업(4대보험 취득)이 필요한 형태인가?"),
    ("기타",                 "진단항목38", "현재 병가 또는 개인 질병 휴직중인 상태인가?"),
    ("기타",                 "진단항목39", "현재 남성 모성제도 사용중인가? (배우자 출산휴가, 육아휴직 등)"),
    ("기타",                 "진단항목40", "현재 여성 모성제도 사용중인가? (출산휴가, 육아휴직 등)"),
]
FEATURE_COLS = [f"진단항목{i}" for i in range(1, 41)]

# ─────────────────────────────────────────────
# 2. 모델 학습 / 캐시 로드
# ─────────────────────────────────────────────
MODEL_PATH = "trained_models.pkl"

@st.cache_resource
def load_or_train_models():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    try:
        df = pd.read_csv("학습데이터.csv", encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv("학습데이터.csv", encoding="cp949")
    le = LabelEncoder()
    y = le.fit_transform(df["결과"].astype(str))
    allow_class = list(le.classes_).index("허용") if "허용" in le.classes_ else 1
    X = df[FEATURE_COLS].fillna(0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        "Random Forest":      RandomForestClassifier(n_estimators=200, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Neural Network":     MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        "SVM":                SVC(probability=True, random_state=42),
        "Gradient Boosting":  GradientBoostingClassifier(n_estimators=200, random_state=42),
    }
    trained = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        acc = m.score(X_test, y_test)
        trained[name] = {"model": m, "accuracy": round(acc * 100, 1)}
    result = {"models": trained, "le": le, "allow_class": allow_class, "feature_cols": FEATURE_COLS}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(result, f)
    return result

# ─────────────────────────────────────────────
# 3. 판단 근거 생성
# ─────────────────────────────────────────────
def generate_reason(model_name, values, prediction, proba):
    is_ccl = values.get("진단항목8", 0) == 1 or values.get("진단항목9", 0) == 1
    risk_map = {
        "진단항목13": "평일 코어타임 겸업 활동으로 본업 집중도에 영향을 줄 수 있음",
        "진단항목15": "겸업을 위한 간헐적 연차 사용으로 본업 일정 운영에 영향을 줄 수 있음",
        "진단항목16": "겸업을 위해 연차를 정기적으로 활용하는 패턴은 본업 운영에 구조적 부담을 초래할 수 있음",
        "진단항목18": "겸업 활동이 본업 수행 역량에 실질적인 영향을 미칠 가능성이 있음",
        "진단항목19": "겸업 활동의 성격이 회사 이익과 상반되는 방향으로 전개될 여지가 있음",
        "진단항목22": "본업을 통해 획득한 기술이나 지식이 겸업에 직접 활용되는 구조로 이해충돌 소지가 있음",
        "진단항목23": "회사의 원천 기술 또는 영업비밀이 겸업 활동에 연계될 우려가 있음",
        "진단항목24": "동종 업계 또는 경쟁 관계에 있는 업체와의 겸업은 정보 유출 및 이해충돌 위험이 높음",
        "진단항목25": "겸업을 통해 생성되는 결과물이나 정보가 회사에 불이익을 줄 가능성이 있음",
        "진단항목27": "회사 소유 자산이나 시설이 겸업에 사용될 경우 자산 보호 관점에서 문제가 될 수 있음",
        "진단항목36": "실질적 운영자와 명의자가 다른 구조는 투명성 측면에서 허용 기준에 부합하지 않음",
        "진단항목37": "4대보험 취득이 수반되는 이중취업은 취업규칙상 겸업 금지 조항에 직접 저촉됨",
        "진단항목38": "휴직 사유와 겸업 활동 간의 불일치는 휴직 목적에 반하는 행위로 간주될 수 있음",
    }
    safe_map = {
        "진단항목10": "업무시간 외 활동으로 본업과 시간적 충돌 없음",
        "진단항목14": "주말 중심의 활동으로 평일 업무 영향 최소화",
        "진단항목34": "일회성 소득으로 지속적 겸업 구조 아님",
        "진단항목7":  "본업 외 개인 역량 기반의 외부 활동으로 직접적 이해충돌 가능성 낮음",
        "진단항목5":  "일회성 외부 출강으로 정기 겸업과 구분됨",
        "진단항목8":  "CCL(크리에이티브 챌린저스 리그) 소속 활동으로 회사 승인 프로그램 해당",
        "진단항목9":  "CCL 연계 부속활동으로 회사 인정 범위 내 활동",
    }
    # 실제 체크된 항목 기준으로만 필터링
    checked_keys = {k for k, v in values.items() if v == 1}
    detected_risks = [desc for k, desc in risk_map.items() if k in checked_keys]
    detected_safe  = [desc for k, desc in safe_map.items() if k in checked_keys]
    risk_cnt = len(detected_risks)
    safe_cnt = len(detected_safe)

    if is_ccl and prediction == 1:
        base = "CCL(크리에이티브 챌린저스 리그)은 회사가 운영하는 공식 프로그램으로 원칙적으로 허용 범주에 해당함"
        detail = "\n· ".join(detected_safe) if detected_safe else "주요 위험 요인 미해당"
        return f"{base}\n· {detail}"

    if prediction == 0:
        base_map = {
            "Random Forest":      f"다수의 결정 트리 분석 결과, 핵심 위험 지표 {risk_cnt}개가 비허용 패턴과 일치",
            "Logistic Regression": f"위험 변수들의 누적 가중치가 허용 임계값을 초과 (감지된 위험지표 {risk_cnt}개)",
            "Neural Network":     f"복합 입력 패턴 분석 결과 비허용 확률 {proba:.0%} 산출",
            "SVM":                f"결정 경계 분석 결과 비허용 영역으로 분류 (위험지표 {risk_cnt}개 반영)",
            "Gradient Boosting":  "단계적 부스팅 분석 결과 비허용 특성 패턴이 강하게 감지됨",
        }
        base = base_map.get(model_name, "비허용 패턴 감지")
        detail = "\n· ".join(detected_risks) if detected_risks else "복합적 위험 요인의 조합이 비허용 기준에 해당"
        return f"{base}\n· {detail}"
    else:
        base_map = {
            "Random Forest":      f"다수의 결정 트리 분석 결과 허용 지표 우세 (안전지표 {safe_cnt}개 확인)",
            "Logistic Regression": "위험 변수의 가중합이 허용 범위 내에 있으며 주요 위험 요인 미해당",
            "Neural Network":     f"복합 입력 패턴 분석 결과 허용 확률 {proba:.0%} 산출",
            "SVM":                "결정 경계 분석 결과 허용 영역으로 분류",
            "Gradient Boosting":  "단계적 부스팅 분석 결과 허용 패턴이 우세하게 감지됨",
        }
        base = base_map.get(model_name, "허용 패턴 감지")
        detail = "\n· ".join(detected_safe) if detected_safe else "주요 위험 요인 미해당"
        return f"{base}\n· {detail}"

# ─────────────────────────────────────────────
# 4. 예측 함수
# ─────────────────────────────────────────────
def predict_all(model_data, input_values):
    X = np.array([[input_values[c] for c in FEATURE_COLS]])
    allow_class = model_data["allow_class"]
    rows = []
    for name, info in model_data["models"].items():
        m = info["model"]
        pred = int(m.predict(X)[0])
        proba_arr = m.predict_proba(X)[0]
        allow_prob = proba_arr[allow_class]
        deny_prob  = 1 - allow_prob
        label = "✅ 허용" if pred == allow_class else "❌ 비허용"
        reason = generate_reason(name, input_values, 1 if pred == allow_class else 0, allow_prob)
        rows.append({
            "모델": name, "판정": label,
            "허용 확률": f"{allow_prob:.1%}", "비허용 확률": f"{deny_prob:.1%}",
            "정확도(테스트셋)": f"{info['accuracy']}%",
            "판단 근거": reason,
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# 5. 종합 의견
# ─────────────────────────────────────────────
def get_summary(result_df):
    allow_cnt = (result_df["판정"].str.contains("허용") & ~result_df["판정"].str.contains("비허용")).sum()
    deny_cnt  = 5 - allow_cnt
    if deny_cnt == 5:
        return "🔴 종합 비허용 — 전원 비허용 판정 (매우 높은 위험)", "deny"
    elif deny_cnt >= 4:
        return f"🔴 종합 비허용 위험 높음 — 5개 중 {deny_cnt}개 모델이 비허용 판정", "deny"
    elif deny_cnt == 3:
        return f"🟡 판단 주의 필요 — 5개 중 {deny_cnt}개 모델이 비허용 판정 (심층 검토 권고)", "caution"
    elif deny_cnt == 2:
        return f"🟡 조건부 검토 권고 — 5개 중 {deny_cnt}개 모델이 비허용 판정", "caution"
    elif deny_cnt == 1:
        return f"🟢 종합 허용 가능 — 5개 중 {deny_cnt}개 모델만 비허용 (허용 우세)", "allow"
    else:
        return "🟢 종합 허용 — 전원 허용 판정", "allow"

# ─────────────────────────────────────────────
# 6. Supabase 연결 및 기록 저장/불러오기
# ─────────────────────────────────────────────
@st.cache_resource
def get_supabase() -> Client:
    import os
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    # 인코딩 문제 방지
    url = url.encode("ascii", "ignore").decode("ascii") if isinstance(url, str) else url
    key = key.encode("ascii", "ignore").decode("ascii") if isinstance(key, str) else key
    return create_client(url, key)

def load_records():
    try:
        sb = get_supabase()
        res = sb.table("Records").select("*").order("id", desc=True).execute()
        return res.data if res.data else []
    except Exception as e:
        st.error(f"기록 불러오기 실패: {e}")
        return []

def save_records(record: dict):
    try:
        sb = get_supabase()
        sb.table("Records").insert(record).execute()
        return True
    except Exception as e:
        st.error(f"기록 저장 실패: {e}")
        return False

def delete_all_records():
    try:
        sb = get_supabase()
        sb.table("Records").delete().neq("id", 0).execute()
        return True
    except Exception as e:
        st.error(f"초기화 실패: {e}")
        return False

# ─────────────────────────────────────────────
# 7. UI 메인
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🏢 겸업 심사 AI 예측 시스템</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">HR 담당자용 · 5개 AI 모델 동시 예측 · 겸업 허용/비허용 판단 지원</div>', unsafe_allow_html=True)

# 모델 로드
with st.spinner("AI 모델 준비 중..."):
    try:
        model_data = load_or_train_models()
        st.success("✅ 5개 모델 준비 완료")
    except Exception as e:
        st.error(f"모델 로드 실패: {e}\n\n`학습데이터.csv` 파일이 앱과 같은 폴더에 있는지 확인해주세요.")
        st.stop()

# 탭 구성
tab1, tab2 = st.tabs(["🔍 겸업 심사", "📋 심사 기록 보드"])

# ══════════════════════════════════════════════
# TAB 1 — 겸업 심사
# ══════════════════════════════════════════════
with tab1:
    # 사이드바
    with st.sidebar:
        st.header("📋 심사 기본 정보")
        emp_name    = st.text_input("성명", placeholder="홍길동")
        emp_dept    = st.text_input("부서", placeholder="인사팀")
        emp_pos     = st.text_input("호칭", placeholder="매니저")
        side_job    = st.text_input("겸업 내용", placeholder="유튜브 채널 운영")
        review_date = st.date_input("심사 일자")
        st.markdown("**겸업 기간**")
        col_s, col_e = st.columns(2)
        with col_s:
            period_start = st.date_input("시작일", key="period_start", label_visibility="collapsed")
        with col_e:
            period_end   = st.date_input("종료일", key="period_end",   label_visibility="collapsed")
        if period_start > period_end:
            st.warning("⚠️ 종료일이 시작일보다 앞서 있어요.")
        elif period_start == period_end:
            st.caption("📅 당일 겸업 (시작일 = 종료일)")
        st.divider()
        st.caption("※ 본 시스템은 AI 보조 도구이며\n최종 판단은 HR 담당자가 수행합니다.")

    # 입력폼
    st.markdown("### 📝 40개 진단항목 입력")
    st.caption("각 항목에 대해 **예(해당)** 또는 **아니오(미해당)** 를 선택해주세요.")
    input_values = {}
    current_section = ""
    for (section, col_id, question) in ITEMS:
        if section != current_section:
            current_section = section
            st.markdown(f'<div class="section-header">📌 {section}</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"**{col_id}** {question}")
        with c2:
            val = st.radio(col_id, ["아니오", "예"], index=0, key=col_id,
                           label_visibility="collapsed", horizontal=True)
        input_values[col_id] = 1 if val == "예" else 0

    # 예측 버튼
    st.markdown("---")
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        run = st.button("🔍 5개 모델 동시 예측 실행", use_container_width=True, type="primary")

    if run:
        with st.spinner("AI 모델 분석 중..."):
            result_df = predict_all(model_data, input_values)
            summary_text, summary_type = get_summary(result_df)

        st.markdown("---")
        st.markdown("## 📊 예측 결과")

        # 기본 정보 (작은 바 형태)
        st.markdown(f"""
        <div class="info-bar">
            <span><b>성명</b>{emp_name or '-'}</span>
            <span><b>부서</b>{emp_dept or '-'}</span>
            <span><b>호칭</b>{emp_pos or '-'}</span>
            <span><b>겸업내용</b>{side_job or '-'}</span>
            <span><b>겸업기간</b>{period_start} ~ {period_end}</span>
        </div>
        """, unsafe_allow_html=True)

        # 종합 의견
        if summary_type == "deny":
            st.markdown(f'<div class="deny-box">{summary_text}</div>', unsafe_allow_html=True)
        elif summary_type == "caution":
            st.markdown(f'<div class="caution-box">{summary_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="allow-box">{summary_text}</div>', unsafe_allow_html=True)

        # 모델별 결과 테이블
        st.markdown("### 🤖 모델별 판정 결과")
        display_df = result_df[["모델", "판정", "허용 확률", "비허용 확률", "정확도(테스트셋)"]].copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # 판단 근거
        st.markdown("### 💬 모델별 판단 근거")
        for _, row in result_df.iterrows():
            icon = "🟢" if "비" not in row["판정"] else "🔴"
            with st.expander(f"{icon} {row['모델']} — {row['판정']}  ({row['허용 확률']} 허용 확률)"):
                st.text(row["판단 근거"])

        # 예 응답 요약
        yes_items = [(cid, q) for (_, cid, q) in ITEMS if input_values.get(cid, 0) == 1]
        if yes_items:
            with st.expander(f"📋 '예' 응답 항목 요약 ({len(yes_items)}개)"):
                for cid, q in yes_items:
                    st.markdown(f"- **{cid}**: {q}")

        # 결과 CSV 다운로드
        st.markdown("---")
        export_df = result_df[["모델", "판정", "허용 확률", "비허용 확률", "판단 근거"]].copy()
        export_df.loc[len(export_df)] = ["[종합]", summary_text, "", "", ""]
        csv_bytes = export_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("📥 결과 CSV 다운로드", data=csv_bytes,
                           file_name=f"겸업심사결과_{emp_name or '미입력'}_{review_date}.csv",
                           mime="text/csv")

        # ── 심사 기록 저장 폼 ──
        st.markdown("---")
        st.markdown("## 📋 최종 판단 기록")
        st.caption("AI 예측 결과를 참고하여 HR 담당자가 최종 판단을 입력 후 저장해주세요.")
        with st.form("record_form"):
            rc1, rc2 = st.columns([1, 1])
            with rc1:
                final_decision = st.radio("최종 결과",
                    options=["✅ 허용", "❌ 비허용", "🟡 조건부 허용"], index=0)
            with rc2:
                reviewer = st.text_input("심사담당자 성명", placeholder="김인사")
            remark = st.text_area("비고", placeholder="예) 겸업 기간 한정 허용, 분기별 재심사 조건 등", height=80)
            save_btn = st.form_submit_button("💾 기록 저장", use_container_width=True, type="primary")

        if save_btn:
            new_rec = {
                "review_date": str(review_date),
                "name":        emp_name or "-",
                "title":       emp_pos or "-",
                "job_content": side_job or "-",
                "job_period":  f"{period_start} ~ {period_end}",
                "ai_result":   summary_text,
                "final_result": final_decision,
                "remark":      remark or "-",
                "reviewer":    reviewer or "-",
            }
            if save_records(new_rec):
                st.success("✅ 기록이 저장되었습니다! '심사 기록 보드' 탭에서 확인하세요.")

# ══════════════════════════════════════════════
# TAB 2 — 심사 기록 보드
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 📑 누적 심사 기록")
    records = load_records()

    if not records:
        st.info("아직 저장된 심사 기록이 없습니다. 겸업 심사 탭에서 기록을 저장해주세요.")
    else:
        records_df = pd.DataFrame(records)
        # 영문 컬럼 → 한글 표시로 변환
        col_rename = {
            "review_date":  "심사일자",
            "name":         "성명",
            "title":        "호칭",
            "job_content":  "겸업내용",
            "job_period":   "겸업기간",
            "ai_result":    "AI심사종합결과",
            "final_result": "최종결과",
            "remark":       "비고",
            "reviewer":     "심사담당자",
        }
        col_order = list(col_rename.keys())
        records_df = records_df[[c for c in col_order if c in records_df.columns]]
        records_df = records_df.rename(columns=col_rename)
        st.dataframe(records_df, use_container_width=True, hide_index=True)

        # 통계 요약
        st.markdown("---")
        st.markdown("#### 📊 심사 통계")
        stat_cols = st.columns(3)
        with stat_cols[0]:
            st.metric("전체 심사 건수", len(records_df))
        with stat_cols[1]:
            allow_n = records_df["최종결과"].str.contains("허용").sum()
            st.metric("허용", allow_n)
        with stat_cols[2]:
            deny_n = records_df["최종결과"].str.contains("비허용").sum()
            st.metric("비허용", deny_n)

        # 전체 기록 다운로드
        st.markdown("---")
        rec_csv = records_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("📥 전체 기록 CSV 다운로드", data=rec_csv,
                           file_name="겸업심사_기록부.csv", mime="text/csv")

        # 기록 초기화
        if st.button("🗑️ 전체 기록 초기화", type="secondary"):
            if delete_all_records():
                st.success("초기화 완료!")
                st.rerun()
