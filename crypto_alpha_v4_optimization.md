# Phase 4: 검증 통과 전략 최적화

## 배경 & 목표

v3 파이프라인에서 1,164개 전략 중 **5중 검증**을 거쳐 생존한 전략들이 있다.
이 전략들의 트레이드 레벨 약점을 진단하고, 메커니즘 기반으로 개선한 뒤,
**동일한 IS/OOS/WF/랜덤 검증 프레임워크**에서 재검증한다.

**핵심 원칙: "고치는 것"이지 "새로 만드는 것"이 아니다.**
기존 시그널 로직은 유지하되, entry timing·exit rule·filter·position sizing을 개선한다.
개선된 전략이 원본보다 OOS에서 **나빠지면 원본을 유지**한다.

---

## 대상 전략 (7개)

기존 v3 코드를 그대로 사용하되, `src/strategies_v4/` 디렉토리에 개선 버전을 만든다.

| 원본 ID | Tier | 핵심 구조 | OOS ret | 주요 약점 |
|---------|------|----------|---------|----------|
| SEQ_S1.11_S+S4.04_L | S | NR7 숏→NATR반전 롱 | +45.0% | SL 52%, BTC 약세, 횡보장 취약 |
| S2.08_L_aggr_F9 | A | Vortex 크로스 롱, BTC필터 | +16.6% | 파라미터 민감, 거래수 소 |
| S2.08_L_aggr_F3 | A | Vortex 크로스 롱, 횡보필터 | +11.8% | 거래수 소 (105건) |
| S3.12_S_aggr_F8 | B | Pivot S1 이탈 숏, Breadth필터 | +43.6% | SL 51%, 랜덤 미달(p=.069) |
| S1.11_S_aggr | B | NR7 숏 단독 | +24.1% | WR 41%, 연속손실 14회 |
| SEQ_S1.11_S+S3.01_L | B | NR7 숏→Supertrend 롱 | +38.9% | SL 55%, SHORT 편중(1230/1328) |
| S7.04_S_trail | B | Breadth Collapse 숏, 트레일링 | +14.8% | WR 39%, 거래수 소(170) |

---

## 진단: 트레이드 레벨 약점 분석

### 약점 1: 과다한 스탑로스 발동 (전 전략 공통, 50~55%)

모든 전략에서 절반 이상의 트레이드가 **고정 SL(-5% 또는 -7%)에 걸려 청산**.

**원인 분석:**
- 고정 SL이 변동성 수준을 무시함. 하루 ATR이 5%인 자산(DOGE)과 2%인 자산(BTC)에 같은 -5% SL을 적용하면, DOGE는 정상 등락에도 SL이 터짐.
- 진입 직후 일시적 역행(noise)에 SL이 걸린 뒤, 원래 방향으로 가는 "whipsaw" 패턴.

**→ 처방: ATR 기반 동적 SL/TP (Improvement A)**

### 약점 2: BTC/SOL 저성과 (SEQ_S1.11 기준)

BTC: 72건 중 총 +4.3% (거의 0), SOL: 99건 중 +22.2% (WR 36%).
반면 AVAX(+156%), XLM(+120%), DOGE(+102%)는 강함.

**원인 분석:**
- BTC는 가장 효율적인 시장 — NR7 breakout이 잘 먹히지 않음.
- SOL은 2023 이후 유동성 증가로 비효율성 감소.

**→ 처방: 자산별 가중치 / 자산 선택 (Improvement D)**

### 약점 3: 횡보장 연속 손실 (SEQ_S1.11: 최대 11~14연패)

2024-02(-69.5%), 2022-11(-44.4%), 2023-01(-41.5%) — 횡보/좁은 범위장에서 NR7 브레이크아웃이 반복 실패.

**원인 분석:**
- NR7은 "좁은 범위 → 확장"을 전제하지만, 횡보가 지속되면 계속 가짜 신호.
- ADX가 낮은 구간에서 방향성 전략이 반복 손실.

**→ 처방: 추세 강도 필터 + 횡보 회피 (Improvement C)**

### 약점 4: MaxHold 청산의 미활용 이익 (SEQ_S1.11: MaxHold avg +2.23%)

MaxHold(10일) 종료 시 평균 +2.23%의 미실현 이익이 남아 있음.
SL/TP에 안 걸리고 10일간 유지된 트레이드는 약간의 수익 상태 — 더 보유하면 더 갈 수 있었음.

**→ 처방: MaxHold 연장 + Break-Even Stop (Improvement B)**

### 약점 5: SEQ 전략의 SHORT 편중 (SEQ_S1.11_S+S3.01_L: SHORT 1230 vs LONG 98)

LONG 신호(Supertrend 반전)가 너무 드물게 발생 → 양방향 수익 포착의 취지 미달.

**→ 처방: LONG 진입 조건 완화 / 대안 LONG 시그널 (Improvement E)**

---

## 개선 모듈 설계 (A~G)

각 모듈은 독립적으로 적용/제거 가능하며, 기존 전략 로직과 조합된다.

### Improvement A: ATR 기반 동적 SL/TP

```python
def dynamic_sl_tp(df, entry_idx, direction, atr_period=14, sl_mult=1.5, tp_mult=3.0):
    """
    진입 시점의 ATR로 SL/TP를 동적 결정.
    SL = entry ± sl_mult × ATR(entry_date)
    TP = entry ± tp_mult × ATR(entry_date)
    """
    atr_at_entry = atr(df, atr_period).iloc[entry_idx]
    if direction == 'LONG':
        sl_price = entry_price - sl_mult * atr_at_entry
        tp_price = entry_price + tp_mult * atr_at_entry
    else:
        sl_price = entry_price + sl_mult * atr_at_entry
        tp_price = entry_price - tp_mult * atr_at_entry
    return sl_price, tp_price
```

**변형 테스트:**

| 변형 | SL mult | TP mult | 의미 |
|------|---------|---------|------|
| A1 (표준) | 1.5 | 3.0 | 1:2 리스크/리워드 |
| A2 (타이트) | 1.0 | 2.5 | 빠른 손절 |
| A3 (와이드) | 2.0 | 4.0 | 노이즈 허용, 큰 움직임 포착 |
| A4 (비대칭) | 1.5 | 5.0 | 손절 표준, 익절 크게 — fat tail 포착용 |

**모든 대상 전략에 A1~A4를 각각 적용하여 IS 검증.**

### Improvement B: 동적 MaxHold + Break-Even Stop

**기존**: 고정 10일 또는 14일 MaxHold.
**개선**: 보유 중 일정 수준의 미실현 이익 도달 시 SL을 진입가로 이동(Break-Even) + MaxHold 연장.

```python
def break_even_exit(entry_price, current_price, direction, 
                     be_trigger_pct=0.03,   # +3% 도달 시 BE 활성화
                     max_hold_base=10,
                     max_hold_extended=21):  # BE 활성화 후 최대 21일
    """
    Phase 1: 일반 SL/TP로 운용 (max_hold_base일까지)
    Phase 2: MtM이 +be_trigger_pct 이상이면:
        - SL을 진입가(+슬리피지)로 이동 (Break-Even)
        - MaxHold를 max_hold_extended로 연장
    Phase 3: 이후 트레일링 또는 반대신호 청산
    """
```

**변형:**

| 변형 | BE trigger | 연장일 | 추가 로직 |
|------|-----------|--------|----------|
| B1 | +3% | 21일 | BE 후 고정 SL=진입가 |
| B2 | +3% | 21일 | BE 후 1×ATR 트레일링 활성화 |
| B3 | +5% | 14일 | BE 후 절반 익절(나머지 트레일링) |

### Improvement C: 횡보장 회피 필터 (Whipsaw Guard)

**기존 필터**: F2(하락추세), F3(ADX<20 횡보), F8(Breadth<40%), F9(BTC 동조)
**신규 필터**: 횡보/연속손실을 사전 차단하는 추가 조건.

| ID | 이름 | 조건 | 적용 대상 |
|----|------|------|----------|
| FC1 | **ADX 최소 추세** | ADX(14) > 15 (최소한의 방향성 존재) | NR7 기반 전략 (S1.11) |
| FC2 | **ATR 확장 확인** | ATR(5) > ATR(20) (단기 변동성이 장기 대비 확대 중) | 브레이크아웃 전략 전체 |
| FC3 | **최근 손실 제한** | 해당 전략의 직전 3회 트레이드 중 3회 모두 손실이면 다음 시그널 1회 건너뜀 (cooldown 연장) | 연속손실 14회가 나왔던 S1.11_S_aggr |
| FC4 | **변동성 레짐** | 60일 실현변동성 percentile > 30% (극저변동성 구간 회피) | SEQ 전략 |
| FC5 | **Efficiency Ratio** | ER = abs(close - close[N]) / sum(abs(close[i]-close[i-1]), N) > 0.3 — 가격 이동의 "효율"이 최소 30% (직선적 움직임) | 모멘텀/방향성 전략 전체 |

### Improvement D: 자산별 차등 처리

데이터 기반 관찰:

| 자산 | SEQ_S1.11 총수익 | WR | 판정 |
|------|------------------|----|------|
| AVAX | +156.1% | 48% | ★★★ 최우선 |
| XLM | +119.8% | 51% | ★★★ |
| DOGE | +101.9% | 45% | ★★☆ |
| XRP | +86.8% | 46% | ★★☆ |
| ETH | +47.2% | 48% | ★☆☆ |
| OP | +41.3% | 38% | ★☆☆ |
| SUI | +37.3% | 42% | ★☆☆ |
| SOL | +22.2% | 36% | ☆☆☆ |
| BTC | +4.3% | 44% | ☆☆☆ |

**접근법 2가지 (둘 다 테스트):**

**D1: 자산 필터링** — 하위 2개(BTC, SOL) 제외, 7개 자산만 운용.
**D2: 자산별 가중치** — 전 자산 운용하되, 보유 기간 수익에 자산별 가중치 적용(포트폴리오 시뮬레이션에서):
  - ★★★: 가중치 1.5
  - ★★☆: 가중치 1.0
  - ★☆☆: 가중치 0.5
  - ☆☆☆: 가중치 0.25

⚠️ **과적합 경고**: D1/D2 모두 IS 데이터 기반이므로, OOS에서 BTC/SOL이 오히려 잘 될 수 있음.
반드시 OOS에서 원본 vs D1 vs D2를 비교.

### Improvement E: SEQ 전략의 LONG 비율 개선

**문제**: SEQ_S1.11_S+S3.01_L에서 LONG(Supertrend 반전)이 98/1328 = 7%만 차지.
Supertrend 반전이 너무 드물게 발생하여 양방향 수익 포착의 취지에 미달.

**해결 방향 (3가지 대안 LONG 시그널 테스트):**

| 변형 | LONG 시그널 | 이유 |
|------|-----------|------|
| E1 | S4.04_L (NATR 반전) | 이미 Tier S에서 검증됨 — 다른 SEQ에도 적용 |
| E2 | RSI(14) < 30 크로스업 | 가장 단순한 과매도 반등 시그널. 발동 빈도 높음 |
| E3 | Fisher Transform < -1 에서 반등 | 비주류 시그널, S2.01 기반 |
| E4 | 3일 연속 음봉 후 양봉 (S1.13 변형) | 프라이스 액션 기반, 캔들 패턴 |

각 변형을 기존 SEQ 프레임워크에 넣어 IS 테스트.

### Improvement F: 트레일링 로직 고도화

**기존**: 고정 -3% 트레일링 (S7.04_S_trail에서만 사용).
**개선**: ATR 기반 + 단계별 트레일링.

```python
def stepped_trailing_stop(peak_price, current_atr, direction,
                           base_mult=2.0,       # 기본: 2×ATR
                           tight_mult=1.0,       # 이익 확대 시: 1×ATR
                           tighten_at_pct=0.05): # +5% 이상 수익 시 타이트
    """
    Phase 1: 기본 트레일링 = peak ± base_mult × ATR
    Phase 2: MtM +5% 이상이면 = peak ± tight_mult × ATR (더 타이트)
    """
```

**변형:**

| 변형 | 기본 mult | 타이트 mult | 타이트 조건 |
|------|----------|-----------|-----------|
| F1 | 2.0×ATR | 1.0×ATR | +5% |
| F2 | 2.5×ATR | 1.5×ATR | +7% |
| F3 | 1.5×ATR | 0.7×ATR | +3% (공격적) |

### Improvement G: 포트폴리오 결합

**단일 전략이 아닌, 상위 전략 3~5개를 동시 운용하는 포트폴리오 전략.**

**기본 규칙:**
- 각 전략은 독립적으로 시그널 생성 & 트레이드 실행.
- 같은 자산에 같은 방향 포지션이 2개+ 전략에서 동시 발생하면 1개만 유지(먼저 진입한 것).
- 같은 자산에 반대 방향이 동시 발생하면 **둘 다 무시** (충돌 회피).
- 포트폴리오 수익 = 전 전략의 전 트레이드 PnL 합산 / 전략 수 (균등 배분).

**테스트할 포트폴리오:**

| 포트폴리오 | 구성 전략 | 의도 |
|-----------|----------|------|
| P1 (Top3) | SEQ_S1.11+S4.04, S2.08_F9, S3.12_F8 | Tier S + Tier A + Tier B 혼합 |
| P2 (방향 분산) | SEQ_S1.11+S4.04, S2.08_F3, S7.04_trail | 숏+롱+트레일링 분산 |
| P3 (SEQ only) | SEQ_S1.11+S4.04, SEQ_S1.11+S3.01, SEQ_S5.02+S3.01 | 양방향 SEQ만 |
| P4 (전 Tier A+B) | 7개 전체 | 최대 분산 |

---

## 실행 계획

### Step 1: 개선 전략 생성

각 대상 전략(7개) × 각 Improvement 모듈 조합:

```
원본 전략 7개
× Improvement A (4 변형: A1~A4)     = 28개
× Improvement B (3 변형: B1~B3)     = 21개
× Improvement C (5 필터: FC1~FC5)   = 35개
× Improvement D (2 변형: D1~D2)     = 14개
× Improvement E (4 변형, SEQ만 적용) = 12개 (SEQ 3개 × 4)
× Improvement F (3 변형)            = 21개
= 소계: ~131개 단일 모듈 적용 전략

+ 유망 조합 (A+B, A+C, A+C+D 등): ~50개
+ 포트폴리오 (P1~P4): 4개
= 총 ~185개 개선 후보
```

### Step 2: IS 백테스트 (2022-01 ~ 2024-06)

185개 전체를 IS 구간에서 백테스트.
**원본 전략과 동일 조건** (슬리피지 0.1%, non-overlapping, cooldown 3일).

**비교 기준**: 원본 전략의 IS 성과 대비 개선 여부.
- IS에서 원본보다 나빠진 개선안은 즉시 탈락.
- IS에서 원본보다 나은 것만 OOS로 보냄.

### Step 3: OOS 백테스트 (2024-07 ~ 2026-06)

IS 통과 개선안을 OOS에서 백테스트.
**핵심 비교: 원본 OOS 성과 vs 개선 OOS 성과.**

| 판정 | 조건 |
|------|------|
| ✓ 채택 | OOS Return ↑ AND (Sharpe ↑ OR PF ↑) |
| △ 보류 | OOS Return ↑ BUT Sharpe ↓ OR 트레이드 수 크게 감소 |
| ✗ 기각 | OOS Return ↓ |

### Step 4: Walk-Forward 재검증

OOS 통과 개선안에 대해 5-window Walk-Forward 실행.
원본이 WF 5/5였다면, 개선안도 최소 4/5 이상이어야 채택.

### Step 5: 랜덤 벤치마크

OOS+WF 통과 개선안에 1000회 랜덤 진입 비교.
p < 0.05 유지 확인.

### Step 6: 최종 비교 테이블

```
전략 ID | 원본 OOS ret | 개선 OOS ret | Δ | 원본 Sharpe | 개선 Sharpe | WF | Random p
```

---

## 출력

### 파일 구조

```
project/
├── src/
│   ├── (기존 v3 코드 전부 유지)
│   ├── improvements/
│   │   ├── __init__.py
│   │   ├── dynamic_sl_tp.py       # Improvement A
│   │   ├── break_even.py          # Improvement B
│   │   ├── whipsaw_guard.py       # Improvement C
│   │   ├── asset_weighting.py     # Improvement D
│   │   ├── alt_long_signals.py    # Improvement E
│   │   ├── stepped_trailing.py    # Improvement F
│   │   └── portfolio.py           # Improvement G
│   ├── optimizer.py               # 개선안 생성기 & 비교 엔진
│   └── dashboard_v4.py            # Phase 4 전용 대시보드
├── results_v4/
│   ├── improvement_scores_is.csv
│   ├── improvement_scores_oos.csv
│   ├── comparison_table.csv       # 원본 vs 개선 비교
│   ├── walk_forward_v4.csv
│   ├── random_benchmark_v4.csv
│   ├── portfolio_results.csv
│   ├── june2026_spot_v4.csv
│   └── dashboard_v4.html
└── main_v4.py
```

### 대시보드 차트 (8개)

| # | 차트 |
|---|------|
| 1 | **원본 vs 개선 비교 바 차트** — 각 원본 전략 옆에 최선 개선안의 OOS return 나란히 |
| 2 | **Improvement 모듈별 효과** — A~G 각 모듈이 평균적으로 OOS return을 얼마나 바꿨는지 |
| 3 | **SL 히트율 변화** — 원본(고정 SL) vs A1~A4(ATR SL)의 SL 발동 비율 변화 |
| 4 | **자산별 수익 변화** — 원본 vs D1/D2의 자산별 수익 히트맵 |
| 5 | **Equity Curve 비교** — 원본 Tier S vs 최선 개선안 vs 포트폴리오 P1~P4 |
| 6 | **월별 수익 히트맵** — 원본 vs 최선 개선안 (횡보장 개선 확인) |
| 7 | **IS vs OOS 일관성** — 원본과 개선안의 IS/OOS 산점도 (과적합 감지) |
| 8 | **2026-06 스팟 체크** — 원본 vs 최선 개선안 + 포트폴리오의 6월 트레이드 상세 |

### 터미널 출력

```
══════════════════════════════════════════════════════════════
  Phase 4: Strategy Optimization — Results
══════════════════════════════════════════════════════════════

📊 Improvements tested: ~185 variants of 7 base strategies

🔬 Module effectiveness (avg OOS return change):
  A (ATR SL/TP):        +X.X pp
  B (Break-Even):       +X.X pp
  C (Whipsaw Guard):    +X.X pp
  D (Asset Selection):  +X.X pp
  E (Alt LONG signal):  +X.X pp
  F (Stepped Trailing): +X.X pp

🏆 Best improvements (vs original):
  SEQ_S1.11_S+S4.04_L:
    Original: OOS +45.0%  Sharpe 0.87
    Best v4:  OOS +XX.X%  Sharpe X.XX  [+A3+C1+B2]
    Change:   +XX.Xpp return, +X.XX Sharpe

  S3.12_S_aggr_F8:
    Original: OOS +43.6%  Sharpe 0.72
    Best v4:  OOS +XX.X%  Sharpe X.XX  [+A1+F1]
    ...

📦 Portfolio results:
  P1 (Top3):     OOS +XX.X%  Sharpe X.XX  MaxDD XX%
  P2 (방향분산): OOS +XX.X%  Sharpe X.XX  MaxDD XX%
  ...

📁 Dashboard: results_v4/dashboard_v4.html
══════════════════════════════════════════════════════════════
```

---

## 핵심 원칙

1. **원본 대비 개선만 채택**: OOS에서 원본보다 나빠지면 기각. "IS에서 좋아졌다"는 근거가 아니다.
2. **검증 프레임워크 동일**: 슬리피지·cooldown·non-overlapping·비복리 합산 모두 v3와 동일.
3. **개선 모듈 독립성**: 각 모듈이 단독 효과를 확인한 후에 조합. "A+B+C+D 한꺼번에" 테스트는 금지 — 어떤 모듈이 기여했는지 분리 불가.
4. **포트폴리오는 마지막**: 개별 전략 최적화 완료 후 포트폴리오 결합. 순서 중요.
5. **Look-ahead bias 금지**: ATR 기반 SL도 진입 시점의 ATR만 사용. 미래 ATR 참조 금지.
6. **과적합 경각심**: Improvement D(자산 필터링)는 IS 데이터 기반이므로 OOS에서 반전 가능. 겸손하게 해석.

---

## 실행

```bash
python main_v4.py                  # 전체 최적화 파이프라인
python main_v4.py --module A       # Improvement A만 테스트
python main_v4.py --portfolio      # 포트폴리오만 테스트
python main_v4.py --compare        # 원본 vs 최선 비교표만 출력
```

---

## 부록: 기존 v3 코드 활용 가이드

`src/backtester_v3.py`의 `run_backtest()` 함수를 확장한다:

```python
# 기존 인터페이스
def run_backtest(strategy, df, symbol, stop_loss=-0.05, take_profit=0.10, 
                 slippage=0.001, cooldown=3) -> list[Trade]

# v4 확장 — 동적 SL/TP, break-even, stepped trailing 지원
def run_backtest_v4(strategy, df, symbol,
                     sl_mode='fixed',        # 'fixed' | 'atr'
                     sl_param=-0.05,         # fixed: pct, atr: multiplier
                     tp_mode='fixed',        # 'fixed' | 'atr'
                     tp_param=0.10,          # fixed: pct, atr: multiplier
                     trailing_mode='none',   # 'none' | 'fixed' | 'atr' | 'stepped'
                     trailing_param=None,
                     break_even=None,        # None | {'trigger': 0.03, 'extend_to': 21}
                     extra_filters=None,     # list of filter functions
                     asset_weight=1.0,       # 포트폴리오 가중치
                     slippage=0.001,
                     cooldown=3) -> list[Trade]
```

기존 v3 전략 클래스의 `entry_signal()`, `exit_signal()`은 그대로 재사용.
개선 모듈은 **exit 처리와 필터에만 개입**하여 시그널 로직은 건드리지 않는다.
