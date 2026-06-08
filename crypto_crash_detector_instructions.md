# 크립토 대폭락 선행신호 탐색 & 백테스트 프로젝트

## 프로젝트 배경

2026년 6월 4일 전후, 크립토 시장에 대규모 하락이 발생했다.
BTC가 ~$66,800 → ~$60,000 (약 -10%), ETH/SOL/알트코인은 그 이상 하락.
Strategy(구 MicroStrategy)의 BTC 매도, ETF 자금 유출, 미-이란 지정학 리스크, 레버리지 청산 캐스케이드가 복합 작용.

**핵심 질문**: "이런 대폭락 직전에 반복적으로 나타나는 기술적 신호가 있는가?"

---

## 목표

1. OHLCV 데이터만으로 계산 가능한 **선행 신호 후보 30~40개**를 체계적으로 정의
2. 2022~2026.05 데이터에서 **"의미 있는 하락 이벤트"**를 객관적으로 식별
3. 각 신호가 하락 이벤트를 **사전에** 얼마나 잘 포착했는지 백테스트
4. 개별 성적표 + 최적 조합(앙상블) 탐색
5. 결과를 **시각화** (인터랙티브 HTML 대시보드)

---

## 데이터

### 대상 자산 (9종)
BTC, ETH, SOL, DOGE, OP, AVAX, XRP, XLM, SUI

### 데이터 구조
- 기간: 2022-01-01 ~ 2026-05-31 (자산별로 상장 이후부터)
- 타임프레임: **일봉 (Daily)**
- 컬럼: `date, open, high, low, close, volume`
- 파일 위치: 프로젝트 루트의 `data/` 디렉토리
  - 파일명 패턴: `{SYMBOL}_daily_ohlcv.csv` (예: `BTC_daily_ohlcv.csv`)
  - 또는 통합 파일 `all_ohlcv.csv`에 `symbol` 컬럼 포함
- **데이터가 없을 경우**: ccxt 라이브러리로 Binance에서 직접 다운로드하는 스크립트를 먼저 작성할 것
  - `pip install ccxt pandas`
  - Binance 퍼블릭 API (API 키 불필요, OHLCV는 공개 엔드포인트)
  - 페어: `{SYMBOL}/USDT`
  - SUI는 2023-05 이후, OP는 2022-06 이후 등 상장일 고려

### 데이터 전처리
- 결측값 처리 (forward fill, 단 5일 이상 연속 결측은 해당 구간 제외)
- 이상치 감지 (일일 수익률 ±50% 초과 시 플래그)
- 거래량 0인 날 제외

---

## Phase 1: 하락 이벤트 정의

### 1.1 하락 이벤트 식별 기준

**다중 기준으로 정의** (하나라도 충족 시 "이벤트"):

| 기준 ID | 정의 | 설명 |
|---------|------|------|
| CRASH_A | 7일 rolling 수익률 ≤ -15% | 1주일 내 급락 |
| CRASH_B | 3일 rolling 수익률 ≤ -10% | 3일 내 급락 |
| CRASH_C | 고점 대비 drawdown ≥ 20% (52주 고점 기준) | 구조적 하락 |
| CRASH_D | 일일 수익률 ≤ -8% | 단일 일 폭락 |

### 1.2 이벤트 클러스터링
- 같은 하락 국면에 속하는 이벤트들을 하나로 묶음
- 이벤트 간 간격이 **14일 이내**면 같은 클러스터
- 각 클러스터의 "시작일"을 해당 하락 국면의 대표 이벤트로 사용

### 1.3 구현

```python
def identify_crash_events(df, symbol):
    """
    Returns DataFrame with columns:
    - date, symbol, crash_type (A/B/C/D), severity (% decline),
    - cluster_id, cluster_start, cluster_peak_decline
    """
    pass
```

### 1.4 출력
- 자산별 하락 이벤트 목록 테이블
- 전체 자산의 "동시 하락" 이벤트 (같은 주에 3개+ 자산이 이벤트 발생)
- 이벤트 히트맵 시각화 (x: 시간, y: 자산, 색: 하락 정도)

---

## Phase 2: 후보 인디케이터 정의 및 계산

### 카테고리 A — 모멘텀 (8개)

| ID | 인디케이터 | 시그널 조건 | 파라미터 |
|----|-----------|------------|---------|
| M1 | RSI(14) 과매수 | RSI > 70 후 70 아래로 하락 | period=14, threshold=70 |
| M2 | RSI 베어리시 다이버전스 | 가격 higher high + RSI lower high | period=14, lookback=20 |
| M3 | MACD 데드크로스 | MACD line이 signal line 하향 돌파 | fast=12, slow=26, signal=9 |
| M4 | MACD 히스토그램 음전환 | 히스토그램 양→음 전환 | fast=12, slow=26, signal=9 |
| M5 | Stochastic %K/%D 크로스 | %K가 %D를 80 위에서 하향 돌파 | k=14, d=3, threshold=80 |
| M6 | CCI 극단값 반전 | CCI > 100 후 100 아래로 하락 | period=20, threshold=100 |
| M7 | Williams %R 과매수 이탈 | %R > -20 후 -20 아래로 하락 | period=14 |
| M8 | ROC 꺾임 | ROC 양수→음수 전환 (고점 이후) | period=10 |

### 카테고리 B — 추세/구조 (7개)

| ID | 인디케이터 | 시그널 조건 | 파라미터 |
|----|-----------|------------|---------|
| T1 | MA 데드크로스 (단기) | EMA(20)이 EMA(50) 하향 돌파 | short=20, long=50 |
| T2 | MA 데드크로스 (장기) | SMA(50)이 SMA(200) 하향 돌파 | short=50, long=200 |
| T3 | 가격 < EMA(200) 이탈 | 종가가 EMA(200) 하향 돌파 | period=200 |
| T4 | ADX 방향 전환 | +DI가 -DI를 하향 돌파 (ADX>25일 때) | period=14, adx_threshold=25 |
| T5 | Parabolic SAR 반전 | SAR이 가격 위로 전환 | af=0.02, max_af=0.2 |
| T6 | Lower High + Lower Low | 최근 고점 < 이전 고점 AND 최근 저점 < 이전 저점 | lookback=20 |
| T7 | 이치모쿠 구름 하향 이탈 | 가격이 구름(선행스팬A, B) 아래로 이탈 | tenkan=9, kijun=26, senkou=52 |

### 카테고리 C — 변동성 (7개)

| ID | 인디케이터 | 시그널 조건 | 파라미터 |
|----|-----------|------------|---------|
| V1 | BB 하방 이탈 | 종가 < 하단밴드(2σ) | period=20, std=2 |
| V2 | BB 폭 Squeeze 후 확장 | BB Width가 6개월 최저 후 급증 | period=20, squeeze_pctile=10 |
| V3 | ATR 급등 | ATR이 20일 평균 대비 1.5배 이상 | period=14, multiplier=1.5 |
| V4 | Garman-Klass 변동성 급등 | GK vol이 20일 이동평균 대비 2σ 초과 | period=20 |
| V5 | 일일 레인지 확대 | (High-Low)/Close가 2σ 초과 | lookback=30 |
| V6 | Keltner Channel 하방 이탈 | 종가 < 하단 Keltner | period=20, atr_mult=2 |
| V7 | 연속 음봉 카운트 | 3일+ 연속 close < open | min_consecutive=3 |

### 카테고리 D — 거래량 (6개)

| ID | 인디케이터 | 시그널 조건 | 파라미터 |
|----|-----------|------------|---------|
| VOL1 | 비정상 거래량 스파이크 | Volume z-score > 2 (하락 일에) | lookback=30, z_threshold=2 |
| VOL2 | OBV 다이버전스 | 가격 higher high + OBV lower high | lookback=20 |
| VOL3 | CMF 음전환 | CMF(20) < 0으로 전환 | period=20 |
| VOL4 | Volume-Price Trend 꺾임 | VPT의 EMA 하향 돌파 | vpt_ema=14 |
| VOL5 | Force Index 음전환 | 13일 EMA Force Index < 0 전환 | period=13 |
| VOL6 | 하락 거래량 비율 급등 | 최근 5일 중 하락일 거래량 / 상승일 거래량 > 2 | period=5, ratio_threshold=2 |

### 카테고리 E — 크로스에셋/복합 (7개)

| ID | 인디케이터 | 시그널 조건 | 파라미터 |
|----|-----------|------------|---------|
| X1 | BTC 대비 약세 | ALT/BTC 비율의 20일 수익률 < -10% | period=20, threshold=-10% |
| X2 | 다자산 동시 RSI 과매수 | 9개 중 5개+ RSI > 70 동시 충족 | count_threshold=5 |
| X3 | 크로스에셋 상관관계 급등 | 30일 rolling 평균 상관관계 > 0.9 | period=30, corr_threshold=0.9 |
| X4 | 시장 Breadth 악화 | EMA(20) 위 자산 비율이 80%→50% 이하 하락 | ema_period=20 |
| X5 | 섹터 동시 약세 | L1(BTC,ETH,SOL,AVAX,SUI), Meme(DOGE), Infra(OP,XRP,XLM) 전 섹터 음수 | period=7 |
| X6 | 거래량 가중 시장 모멘텀 | 전 자산 volume-weighted 평균 수익률 5일 연속 음수 | period=5 |
| X7 | 변동성 체제 전환 | 다자산 평균 ATR이 regime threshold 돌파 | atr_period=14, regime_lookback=60 |

### 구현 가이드

```python
# 각 인디케이터는 통일된 인터페이스를 따를 것
class Indicator:
    def __init__(self, name: str, category: str, params: dict):
        self.name = name
        self.category = category
        self.params = params

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns: pd.Series of signal values (float)
        Index = date
        """
        raise NotImplementedError

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns: pd.Series of boolean signals (True = warning fired)
        Index = date
        """
        raise NotImplementedError
```

**라이브러리**: `ta` (Technical Analysis Library), `pandas_ta`, 또는 직접 구현.
가급적 `pandas_ta`를 우선 사용하되, 없는 것은 직접 구현.

---

## Phase 3: 백테스트 프레임워크

### 3.1 평가 방식

각 인디케이터의 시그널이 **하락 이벤트 N일 전**에 발생했는지 평가.

```
[Signal Window]          [Crash Event]
   ◄── 1~N일 ──►        ▼
───●────────────────────●───────────
 시그널 발생             하락 시작
```

**평가 윈도우**: 시그널 발생 후 1~21일 이내에 하락 이벤트 시작 → True Positive

### 3.2 메트릭

각 인디케이터 × 각 자산 × 각 하락 기준에 대해:

| 메트릭 | 정의 | 의미 |
|--------|------|------|
| **Precision** | TP / (TP + FP) | 시그널이 울렸을 때 실제로 하락이 온 비율 |
| **Recall** | TP / (TP + FN) | 실제 하락 중 사전에 시그널이 잡힌 비율 |
| **F1 Score** | 2 × P × R / (P + R) | 균형 점수 |
| **Avg Lead Time** | 시그널→하락 시작 평균 일수 | 얼마나 미리 경고했는가 |
| **False Alarm Rate** | FP / total signals | 헛경보 비율 |
| **Hit Rate by Crash Type** | 각 CRASH_A/B/C/D별 recall | 어떤 유형의 하락을 잘 잡는가 |

### 3.3 구현

```python
def backtest_indicator(
    indicator: Indicator,
    price_df: pd.DataFrame,
    crash_events: pd.DataFrame,
    forward_window: int = 21,  # 시그널 후 N일 이내 하락 발생 여부
    min_lead_time: int = 1     # 최소 선행 일수 (당일 제외)
) -> dict:
    """
    Returns:
    {
        'precision': float,
        'recall': float,
        'f1': float,
        'avg_lead_time': float,
        'median_lead_time': float,
        'false_alarm_rate': float,
        'total_signals': int,
        'true_positives': int,
        'false_positives': int,
        'missed_crashes': int,
        'signal_dates': list,
        'tp_dates': list,
        'fp_dates': list,
    }
    """
    pass
```

### 3.4 파라미터 민감도 분석

상위 10개 인디케이터에 대해 핵심 파라미터를 ±20% 변화시키며 성능 안정성 확인.
예: RSI period = [10, 12, 14, 16, 18], threshold = [65, 70, 75]

---

## Phase 4: 스코어링 & 랭킹

### 4.1 종합 점수 계산

```python
composite_score = (
    0.30 * precision +        # 정확성 가중
    0.25 * recall +            # 포착률
    0.20 * (1 - false_alarm_rate) +  # 헛경보 적음
    0.15 * normalized_lead_time +     # 선행성
    0.10 * consistency_across_assets  # 다자산 일관성
)
```

**consistency_across_assets**: 9개 자산 중 F1 > 0.3인 자산의 비율

### 4.2 출력

- 전체 인디케이터 랭킹 테이블 (종합 점수 내림차순)
- 카테고리별 Best Performer
- 자산별 Best Performer
- 하락 유형별 Best Performer (CRASH_A~D)

---

## Phase 5: 앙상블 탐색

### 5.1 조합 전략

상위 10개 인디케이터를 2~4개씩 조합:

| 전략 | 룰 |
|------|-----|
| AND (보수적) | 모든 인디케이터가 동시에 시그널 → 경고 |
| MAJORITY | 과반수가 시그널 → 경고 |
| OR (공격적) | 하나라도 시그널 → 경고 |
| WEIGHTED | 각 인디케이터에 개별 F1 기반 가중치 → 가중합 > threshold |
| SEQUENTIAL | 인디케이터 A 발생 후 N일 내 인디케이터 B 발생 → 경고 |

### 5.2 조합 수 제한

- 상위 10개 중 2개 조합: C(10,2) = 45
- 상위 10개 중 3개 조합: C(10,3) = 120
- 3개 전략 × 165 조합 = ~500 백테스트 → 충분히 실행 가능
- 전체 조합 탐색 후 상위 10개 앙상블 보고

---

## Phase 6: 출력 & 시각화

### 6.1 파일 구조

```
project/
├── data/                        # OHLCV 원본
├── src/
│   ├── data_loader.py           # 데이터 로드/전처리
│   ├── crash_detector.py        # Phase 1: 하락 이벤트 식별
│   ├── indicators/
│   │   ├── __init__.py
│   │   ├── base.py              # Indicator 베이스 클래스
│   │   ├── momentum.py          # M1~M8
│   │   ├── trend.py             # T1~T7
│   │   ├── volatility.py        # V1~V7
│   │   ├── volume.py            # VOL1~VOL6
│   │   └── cross_asset.py       # X1~X7
│   ├── backtester.py            # Phase 3: 백테스트 엔진
│   ├── scorer.py                # Phase 4: 스코어링
│   ├── ensemble.py              # Phase 5: 앙상블
│   └── visualizer.py            # Phase 6: 시각화
├── results/
│   ├── crash_events.csv
│   ├── individual_scores.csv
│   ├── ensemble_scores.csv
│   └── dashboard.html           # 인터랙티브 대시보드
├── main.py                      # 전체 파이프라인 실행
└── requirements.txt
```

### 6.2 대시보드 (dashboard.html)

Plotly 또는 Matplotlib으로 인터랙티브 HTML 생성. 반드시 포함할 차트:

1. **이벤트 히트맵**: x=날짜, y=자산, 색=하락 강도, 모든 하락 이벤트 시각화
2. **인디케이터 랭킹 바 차트**: 종합 점수 내림차순, 카테고리별 색 구분
3. **Precision-Recall 스캐터**: 각 인디케이터를 점으로, 크기=lead time
4. **상위 5개 인디케이터 시계열 오버레이**: BTC 가격 + 시그널 발생 시점 마커 + 실제 하락 구간 음영
5. **앙상블 성능 비교**: 상위 10개 조합의 F1 vs False Alarm Rate
6. **자산별 히트맵**: x=인디케이터, y=자산, 색=F1 (어떤 인디케이터가 어떤 자산에 잘 작동하는지)
7. **파라미터 민감도 차트**: 상위 3개 인디케이터의 파라미터 vs F1

### 6.3 리포트 출력 (터미널)

실행 완료 시 터미널에 요약 출력:

```
═══════════════════════════════════════════════
  Crypto Crash Detector — Backtest Results
═══════════════════════════════════════════════

📊 Data: 9 assets, 2022-01 ~ 2026-05
💥 Crash Events Found: XX total (YY unique clusters)

🏆 Top 10 Indicators:
 #1  [M2] RSI Bearish Divergence  — F1: 0.72, Precision: 0.68, Lead: 4.2d
 #2  [VOL2] OBV Divergence        — F1: 0.67, Precision: 0.71, Lead: 3.8d
 ...

🔗 Best Ensemble:
 [M2 + VOL2 + V2] WEIGHTED — F1: 0.81, Precision: 0.85, Lead: 5.1d

📁 Full results: results/dashboard.html
═══════════════════════════════════════════════
```

---

## 기술 요구사항

### Python 패키지
```
pandas>=2.0
numpy>=1.24
pandas_ta>=0.3.14
plotly>=5.15
scipy>=1.10
scikit-learn>=1.3
ccxt>=4.0           # 데이터 다운로드용
tqdm>=4.65
```

### 실행

```bash
# 1. 환경 설정
pip install -r requirements.txt

# 2. 데이터 다운로드 (data/ 디렉토리에 파일이 없을 경우)
python src/data_loader.py --download

# 3. 전체 파이프라인 실행
python main.py

# 4. 결과 확인
# → results/dashboard.html 열기
```

### 성능 고려사항
- 인디케이터 계산은 벡터화 연산 사용 (for 루프 최소화)
- 앙상블 탐색 시 조합이 많으므로 tqdm으로 진행 상황 표시
- 중간 결과를 CSV로 저장하여 재실행 시 캐싱 가능하게

---

## 주의사항 & 제약

1. **Look-ahead bias 금지**: 모든 인디케이터는 해당 시점에서 과거 데이터만 사용할 것. 미래 데이터 참조하는 지표는 무의미.
2. **Survivorship bias 인지**: SUI(2023~), OP(2022.06~) 등 전체 기간 데이터가 없는 자산 존재. 성능 비교 시 데이터 가용 기간 명시.
3. **과적합 경고**: 인디케이터 35개 × 파라미터 조합을 탐색하면 우연한 패턴을 잡을 수 있음. 결과 해석 시 "2022~2024 in-sample / 2025~2026.05 out-of-sample" 분할 검증을 반드시 수행.
4. **이것은 트레이딩 시스템이 아님**: 슬리피지, 수수료, 유동성 미반영. 신호의 **통계적 유의성** 탐색이 목적.
5. **크로스에셋 지표(X1~X7)는 자산별이 아닌 시장 전체 레벨**: 별도 처리 필요.

---

## 체크리스트

작업 완료 후 다음을 확인:

- [ ] 9개 자산 전체 OHLCV 데이터 로드 성공
- [ ] 하락 이벤트 식별 완료 & crash_events.csv 저장
- [ ] 35개 인디케이터 전체 구현 & 시그널 생성
- [ ] 백테스트 실행 완료 & individual_scores.csv 저장
- [ ] In-sample / Out-of-sample 분할 검증 수행
- [ ] 상위 10개 인디케이터 파라미터 민감도 분석
- [ ] 앙상블 조합 탐색 완료 & ensemble_scores.csv 저장
- [ ] dashboard.html 생성 (7개 차트 포함)
- [ ] 터미널 요약 출력 정상 동작
- [ ] main.py 단일 실행으로 전체 파이프라인 재현 가능

---

## 실행 순서 요약

```
Step 1: 데이터 준비         → data_loader.py
Step 2: 하락 이벤트 식별     → crash_detector.py
Step 3: 인디케이터 계산      → indicators/*.py
Step 4: 백테스트 실행        → backtester.py
Step 5: 스코어링 & 랭킹     → scorer.py
Step 6: In/Out-sample 검증  → scorer.py (split mode)
Step 7: 파라미터 민감도      → scorer.py (sensitivity mode)
Step 8: 앙상블 탐색          → ensemble.py
Step 9: 시각화 & 리포트      → visualizer.py
Step 10: 전체 실행           → main.py
```

하나의 Phase가 끝날 때마다 중간 결과를 출력하고, 다음 Phase로 넘어가기 전에 데이터 무결성을 확인할 것.
