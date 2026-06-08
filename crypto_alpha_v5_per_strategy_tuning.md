# Phase 5: 전략별 심층 파라미터 최적화

## 데이터에서 발견된 핵심 패턴

전 전략에 공통적으로 나타나는 **치명적 패턴**이 있다:

```
보유 0일 트레이드: WR 7~25%, 평균 -1.5~-5.6%
보유 1일 트레이드: WR 13~40%, 평균 -4.3~+0.9%
보유 4~5일:       WR 48~57%, 평균 +2.4~+3.8%
보유 10일+:       WR 62~75%, 평균 +3.7~+4.3%
```

**0~1일 안에 스탑로스에 걸리는 트레이드가 전체의 30~40%를 차지하며,
이것들이 수익의 대부분을 갉아먹고 있다.**

또한 전략별로 **특정 자산이 극단적 손실원**인 경우가 있다:
- S1.11_S_aggr: DOGE -68.8% (유일한 적자 자산)
- SEQ_S1.11_S+S3.01_L: XRP -65.7%
- S3.12_S_aggr_F8: BTC -3.4%, XRP -7.3%

---

## 실행 방법론

각 전략에 대해 아래 **3단계 최적화**를 독립적으로 수행한다.

### Step 1: 파라미터 그리드 서치 (IS 구간)

전략별 맞춤 파라미터 그리드를 IS(2022-01~2024-06)에서 탐색.
그리드의 각 조합마다 IS 백테스트 → IS 메트릭 기록.

### Step 2: IS 상위 선별 → OOS 검증

IS 상위 10개 파라미터 셋을 OOS(2024-07~2026-06)에서 검증.
**원본 및 Phase 4 최선보다 OOS에서 나은 것만 채택.**

### Step 3: Walk-Forward + 랜덤 검정

OOS 채택 시 WF 5-window + 1000회 랜덤 벤치마크 실행.

---

## 전략 1: SEQ_S1.11_S+S4.04_L (Tier S)

### 진단

| 관찰 | 수치 | 시사점 |
|------|------|--------|
| 0일 보유 트레이드 | 170건, WR 25%, avg -1.5% | 진입 직후 즉시 역행 → SL이 초기 노이즈에 취약 |
| 4~5일 보유 | 83건, WR 57%, avg +3.3% | 며칠 버틴 트레이드는 대부분 수익 |
| SOL SL율 63%, BTC 44% | 자산별 편차 큼 | 변동성 높은 자산에 고정 SL 부적합 |
| 손실 월 SL율 63% vs 이익 월 46% | 횡보장에서 SL 연쇄 발동 | 레짐 인식 필요 |
| SHORT 545건 vs LONG 173건 | 3:1 불균형 | LONG 레그 활성화 여지 |

### 파라미터 그리드

```python
SEQ_S1_GRID = {
    # === 진입 관련 ===
    'entry_delay': [0, 1],           
    # 0 = 기존(시그널 다음날 시가)
    # 1 = 시그널 발생 후 1일 대기, 2일째 시가 진입
    #     (0일 보유 -1.5% 문제 완화: 초기 노이즈 지나간 후 진입)
    
    'nr7_lookback': [5, 7, 9],       
    # NR7의 N값. 기존 7. 
    # 5 = 더 빈번한 시그널 (5일 중 최소범위)
    # 9 = 더 희소하지만 더 강한 squeeze
    
    # === 청산 관련 ===
    'sl_mode': ['fixed', 'atr'],     
    'sl_value': {
        'fixed': [-0.03, -0.05, -0.07],  
        # 기존 -0.05. -0.03은 더 빠른 손절, -0.07은 더 넓은 허용
        'atr': [1.0, 1.5, 2.0, 2.5],     
        # ATR 배수. 자산별 변동성에 맞춰 동적 조정
    },
    
    'tp_mode': ['fixed', 'atr'],     
    'tp_value': {
        'fixed': [0.07, 0.10, 0.15],
        'atr': [2.5, 3.0, 4.0, 5.0],    
        # A4(1.5/5.0)가 Phase 4에서 +78%를 만들었음 → 비대칭 중심 탐색
    },
    
    'max_hold': [7, 10, 14, 21],     
    # 기존 10. MaxHold 평균 +2.2% → 연장 시 추가 수익 가능
    
    'trailing': [None, 'atr_stepped'],
    'trailing_params': {
        'atr_stepped': {
            'base_mult': [1.5, 2.0, 2.5],
            'tight_mult': [0.7, 1.0, 1.5],
            'tighten_at': [0.03, 0.05, 0.07],
        }
    },
    
    # === 필터 ===
    'min_atr_ratio': [None, 1.0, 1.2],
    # ATR(5)/ATR(20) > threshold. 
    # 단기 변동성이 장기 대비 확대 중일 때만 진입
    # (0일 보유 손실 = 변동성 없는데 브레이크아웃 베팅 → 필터링)
    
    'cooldown': [3, 5],              
    # 기존 3일. 5일로 늘리면 연속손실 감소 기대
}
```

**그리드 크기 제한**: 
전수 탐색하면 조합이 수천 개. 아래 우선순위로 단계적 탐색:

1단계 (핵심 3개, ~48 조합): `sl_mode×sl_value × tp_mode×tp_value × max_hold`
2단계 (1단계 상위 5개 × 진입 변형, ~30): `× entry_delay × nr7_lookback`
3단계 (2단계 상위 3개 × 트레일링, ~27): `× trailing_params`
4단계 (3단계 상위 2개 × 필터, ~12): `× min_atr_ratio × cooldown`

**총 ~117 조합. IS 탐색 후 상위 10개 → OOS.**

### 특별 변형: 비대칭 ATR 중심 탐색

Phase 4에서 A4(SL 1.5×ATR, TP 5×ATR)가 +78.2%를 만든 점을 고려,
비대칭 ATR을 중심으로 미세 조정:

```python
ASYMMETRIC_GRID = {
    'sl_atr_mult': [1.0, 1.25, 1.5, 1.75, 2.0],
    'tp_atr_mult': [3.0, 4.0, 5.0, 6.0, 7.0],
    'max_hold': [10, 14, 21],
    'entry_delay': [0, 1],
}
# 5 × 5 × 3 × 2 = 150 조합 → IS 탐색 → 상위 10 → OOS
```

---

## 전략 2: S3.12_S_aggr_F8 (Tier B, 6월 최강)

### 진단

| 관찰 | 수치 | 시사점 |
|------|------|--------|
| **10일+ 보유: WR 75%, avg +4.3%** | 114건 | **이 전략은 장기 보유할수록 강하다** |
| 0~1일 보유: WR 21~23%, avg -2.1~-2.6% | 114건 | 초기 청산이 수익을 갉아먹음 |
| 손실 월 SL율 **86%** | 극단적 | SL이 거의 모든 손실의 원인 |
| BTC -3.4%, XRP -7.3% | 유이한 적자 | 이 2개 제외 시 성과 급등 가능 |
| Pivot S1 + Breadth<40% | 시장 약세 확인 후 숏 | 약세장에서만 활성화 → 강세장 방어 불필요 |

### 파라미터 그리드

```python
S312_GRID = {
    # === 핵심: 보유 기간 대폭 연장 ===
    'max_hold': [14, 21, 28, 42],    
    # 기존 14(aggr). 10d+ WR 75% → 장기 보유 시 수익 극대화
    
    # === SL: 초기 SL 완화 ===
    'sl_mode': ['fixed', 'atr'],
    'sl_value': {
        'fixed': [-0.05, -0.07, -0.10, -0.12],
        # 기존 -0.07(aggr). -0.10~-0.12까지 넓히면 0~1일 SL 감소
        'atr': [1.5, 2.0, 2.5, 3.0],
    },
    
    # === TP: 트레일링으로 대체 ===
    'tp_mode': ['fixed', 'trailing_only'],
    'tp_value': {
        'fixed': [0.10, 0.15, 0.20, 0.25],
        # 기존 0.15(aggr). WR 75%@10d+ → TP를 높이거나 제거
        'trailing_only': [None],
        # TP 없이 트레일링만 사용 (수익 상한 제거)
    },
    
    'trailing': ['none', 'atr_stepped', 'atr_simple'],
    'trailing_params': {
        'atr_simple': {'mult': [2.0, 2.5, 3.0]},
        'atr_stepped': {
            'base_mult': [2.0, 2.5, 3.0],
            'tight_mult': [1.0, 1.5],
            'tighten_at': [0.05, 0.07, 0.10],
        }
    },
    
    # === Pivot 기간 ===
    'pivot_mode': ['daily', 'weekly'],
    # daily = 전일 HLC 기반 PP/S1 (기존)
    # weekly = 전주 HLC 기반 → 더 의미 있는 지지선, 시그널 빈도 감소
    
    # === Breadth 문턱 ===
    'breadth_threshold': [0.30, 0.40, 0.50],
    # 기존 0.40. 0.30이면 더 엄격(확실한 약세만), 0.50이면 더 빈번
    
    'entry_delay': [0, 1],
    'cooldown': [3, 5, 7],
}
```

**탐색 전략:**
1단계: `sl × max_hold × tp_mode` (~48)
2단계: `상위5 × trailing × trailing_params` (~45)
3단계: `상위3 × pivot_mode × breadth × entry_delay` (~36)
**총 ~129 조합**

### 핵심 가설: "TP 제거 + 트레일링"

이 전략은 10d+ 보유 시 WR 75%로, 수익 트레이드가 더 갈 수 있는데 TP(+15%)에 막힘.
**TP를 제거하고 ATR 트레일링만 사용하면**, 큰 하락 추세에서 30~50%까지 탈 수 있을 가능성.
6월 하락(-17% BTC)에서 TP가 아니라 트레일링이었으면 더 많이 먹었을 것.

---

## 전략 3: S2.08_L_aggr_F3 (Tier A, p=0.001)

### 진단

| 관찰 | 수치 | 시사점 |
|------|------|--------|
| 총 105건 (작은 표본) | 자산당 4~17건 | 통계적 신뢰도 낮음 → 과최적화 주의 |
| 0일 보유: WR 25%, avg -1.7% | 8건 | 작지만 같은 패턴 |
| 4~10일 보유: WR 50~54% | 44건 | 최적 보유구간 |
| 손실 월 SL율 **92%** | 극단적 | Vortex 잘못 발동 시 거의 항상 SL |
| ADX<20 횡보에서만 활성화 | F3 필터 | 추세 초입을 노리지만 가짜 신호 많음 |

### 파라미터 그리드

```python
S208_F3_GRID = {
    # === Vortex 기간 ===
    'vortex_period': [10, 14, 18, 21],
    # 기존 14. 더 긴 기간 = 더 안정적 크로스, 더 늦은 진입
    
    # === ADX 필터 문턱 ===
    'adx_threshold': [15, 20, 25],
    # 기존 20(F3). 15 = 더 넓은 허용, 25 = 약간의 추세에서도 진입
    
    # === 청산 ===
    'sl_mode': ['fixed', 'atr'],
    'sl_value': {
        'fixed': [-0.05, -0.07, -0.10],
        'atr': [1.5, 2.0, 2.5],
    },
    'tp_mode': ['fixed', 'atr'],
    'tp_value': {
        'fixed': [0.10, 0.15, 0.20],
        'atr': [3.0, 4.0, 5.0],
    },
    'max_hold': [10, 14, 21, 28],
    # 기존 14(aggr). 4~10d가 최적 → 10~14 유지, 
    # 단 트레일링 사용 시 21~28까지 확장
    
    'trailing': ['none', 'atr_simple'],
    'trailing_params': {
        'atr_simple': {'mult': [1.5, 2.0, 2.5]},
    },
    
    'entry_delay': [0, 1],
    'cooldown': [3, 5],
    
    # === BTC 필터 변형 ===
    'btc_filter': [None, 'btc_5d_pos', 'btc_20d_pos'],
    # None = F3만 사용 (기존)
    # btc_5d_pos = BTC 5일 수익률 > 0 추가 (F9 개념)
    # btc_20d_pos = BTC 20일 수익률 > 0 (더 장기 추세)
}
```

⚠️ **주의: 표본 105건에서 파라미터 최적화는 과적합 위험이 매우 높다.**
- 그리드를 작게 유지 (1단계 30조합, 2단계 20조합)
- OOS에서 원본 대비 개선이 **작더라도** 안정적이면 채택
- 큰 개선(+30pp 이상)은 오히려 의심

---

## 전략 4: S1.11_S_aggr (Tier B)

### 진단

| 관찰 | 수치 | 시사점 |
|------|------|--------|
| **0일 보유: WR 7%, avg -5.6%** | 54건 | **재앙적. 진입 직후 즉시 역행** |
| 1일 보유: WR 13%, avg -4.3% | 60건 | 여전히 최악 |
| 4~5일+: WR 48~68% | 297건 | 며칠 지나면 수익 전환 |
| DOGE: **-68.8%** | 66건 | 유일한 적자 자산. DOGE 숏이 반복 실패 |
| 최대 연속손실 **14회** | 심리적 한계 | 연속손실 제한 필요 |
| NR7 → 숏 (기존 방향은 하방만) | 단방향 | NR7 발생 후 시가 방향 추종이 나을 수도 |

### 파라미터 그리드

```python
S111_GRID = {
    # === 진입 개선: 0~1일 문제 해결 ===
    'entry_delay': [0, 1, 2],
    # 2 = 시그널 후 2일 대기. 극단적이지만 0~1일 WR 7~13% 회피
    
    'entry_confirmation': [None, 'close_below_open', 'gap_down'],
    # None = 기존 (NR7 다음날 무조건 진입)
    # close_below_open = NR7 다음날 음봉이면 그 다음날 진입
    #   (방향 확인 후 진입 → 0일 역행 감소)
    # gap_down = 다음날 시가 < 전일 종가면 진입
    #   (갭 다운 = 하락 모멘텀 확인)
    
    # === NR 기간 ===
    'nr_period': [5, 7, 9, 11],
    # 기존 7. 더 긴 기간 = 더 강한 squeeze → 더 확실한 브레이크아웃
    
    # === SL: 대폭 완화 ===
    'sl_mode': ['fixed', 'atr'],
    'sl_value': {
        'fixed': [-0.05, -0.07, -0.10, -0.12, -0.15],
        # 기존 -0.07(aggr). 0~1일 WR 7%의 원인이 tight SL
        # -0.12~-0.15까지 넓히면 단기 noise 허용
        'atr': [1.5, 2.0, 2.5, 3.0],
    },
    
    # === TP ===
    'tp_mode': ['fixed', 'atr'],
    'tp_value': {
        'fixed': [0.10, 0.15, 0.20],
        'atr': [3.0, 4.0, 5.0],
    },
    
    'max_hold': [10, 14, 21, 28],
    # 10d+ WR 68% → 길게 보유할수록 좋음
    
    'trailing': ['none', 'atr_stepped'],
    'trailing_params': {
        'atr_stepped': {
            'base_mult': [2.0, 2.5],
            'tight_mult': [1.0, 1.5],
            'tighten_at': [0.05, 0.07],
        }
    },
    
    # === DOGE 제외 옵션 ===
    'exclude_assets': [[], ['DOGE'], ['DOGE', 'BTC']],
    
    # === 연속손실 쿨다운 ===
    'loss_streak_cooldown': [None, 3, 5],
    # N회 연속 손실 후 1회 건너뜀. 14연패 방지.
    
    'cooldown': [3, 5, 7],
}
```

**핵심 가설: "진입을 늦추면 0~1일 문제가 해결된다"**

entry_delay=1 또는 entry_confirmation으로 진입을 1~2일 뒤로 미루면,
noise에 의한 즉시 SL(0~1일, WR 7~13%)을 건너뛸 수 있다.
대신 시그널 빈도가 줄고 최적 진입점을 놓칠 위험.

---

## 전략 5: SEQ_S1.11_S+S3.01_L (Tier B, E1 적용 시 WF 5/5)

### 진단

| 관찰 | 수치 | 시사점 |
|------|------|--------|
| SHORT 1230건 vs LONG 98건 | 12:1 불균형 | LONG 레그 거의 미활성 |
| XRP: **-65.7%** | 154건 | 극단적 손실원 |
| AVAX: +253.8% | 174건 | 극단적 수익원 |
| 0일 보유: WR 17%, avg -2.7% | 252건(19%) | 전체의 1/5이 진입 직후 역행 |
| E1(NATR 롱) 적용 시 WF 5/5 | Phase 4 결과 | LONG 활성화가 견고성 개선 |

### 파라미터 그리드

```python
SEQ_S3_GRID = {
    # === SHORT 레그 (S1.11 NR7) ===
    'short_nr_period': [5, 7, 9],
    'short_entry_delay': [0, 1],
    
    # === LONG 레그 선택 ===
    'long_signal': ['S3.01', 'S4.04'],
    # S3.01 = Supertrend (기존, 드물게 발동)
    # S4.04 = NATR 반전 (E1, Phase 4에서 검증됨)
    
    # S4.04 파라미터 (long_signal=S4.04일 때)
    'natr_period': [10, 14, 20],
    'natr_spike_threshold': [1.5, 2.0, 2.5],
    # NATR이 SMA 대비 N배 이상일 때 반전 롱 시그널
    
    # S3.01 파라미터 (long_signal=S3.01일 때)
    'supertrend_period': [7, 10, 14],
    'supertrend_mult': [2.0, 3.0, 4.0],
    
    # === 공통 청산 ===
    'sl_mode': ['fixed', 'atr'],
    'sl_value': {
        'fixed': [-0.03, -0.05, -0.07],
        'atr': [1.0, 1.5, 2.0],
    },
    'tp_mode': ['fixed', 'atr'],
    'tp_value': {
        'fixed': [0.07, 0.10, 0.15],
        'atr': [2.5, 3.0, 4.0, 5.0],
    },
    'max_hold': [7, 10, 14],
    
    'trailing': ['none', 'atr_simple'],
    
    # === XRP 처리 ===
    'exclude_assets': [[], ['XRP']],
    
    'cooldown': [3, 5],
}
```

**핵심 가설: "S4.04 LONG의 파라미터를 미세 조정하면 WF 5/5를 유지하며 수익 증가"**

E1(S4.04)이 WF 5/5를 만든 핵심은 LONG 발동 빈도 증가.
NATR spike threshold를 조정하면 LONG/SHORT 비율을 더 균형적으로 만들 수 있음.

---

## 구현 규칙

### 백테스트 엔진 확장

기존 v3/v4 `run_backtest()` 엔진에 아래 기능을 추가:

```python
def run_backtest_v5(
    strategy,
    df: pd.DataFrame,
    symbol: str,
    
    # 진입 관련 (신규)
    entry_delay: int = 0,              # 0=기존, 1=1일대기, 2=2일대기
    entry_confirmation: str = None,     # None, 'close_below_open', 'gap_down'
    exclude_assets: list = [],          # 제외 자산 목록
    
    # 청산 관련 (v4에서 일부 존재, 확장)
    sl_mode: str = 'fixed',
    sl_value: float = -0.05,
    tp_mode: str = 'fixed',
    tp_value: float = 0.10,
    max_hold: int = 10,
    
    trailing_mode: str = 'none',       # 'none', 'atr_simple', 'atr_stepped'
    trailing_base_mult: float = 2.0,
    trailing_tight_mult: float = 1.0,
    trailing_tighten_at: float = 0.05,
    
    # 리스크 관련
    cooldown: int = 3,
    loss_streak_cooldown: int = None,   # N연패 후 1회 스킵
    
    slippage: float = 0.001,
) -> list[Trade]:
```

### entry_delay 구현

```python
if entry_delay > 0:
    # 시그널 발생일 + entry_delay일의 시가에 진입
    # 예: entry_delay=1이면, 시그널 다음날이 아닌 그 다음날 시가
    entry_bar_idx = signal_bar_idx + 1 + entry_delay
    entry_price = df.iloc[entry_bar_idx]['open']
```

### entry_confirmation 구현

```python
if entry_confirmation == 'close_below_open':
    # 시그널 다음날이 음봉(close < open)이면 진입 확인
    # 그 다음날 시가에 진입
    confirm_bar = df.iloc[signal_bar_idx + 1]
    if confirm_bar['close'] < confirm_bar['open']:  # SHORT 확인
        entry_bar_idx = signal_bar_idx + 2
        entry_price = df.iloc[entry_bar_idx]['open']
    else:
        skip  # 확인 실패, 시그널 무시

elif entry_confirmation == 'gap_down':
    # 다음날 시가 < 전일 종가 (갭 다운 확인)
    next_open = df.iloc[signal_bar_idx + 1]['open']
    prev_close = df.iloc[signal_bar_idx]['close']
    if next_open < prev_close:
        entry_price = next_open  # 갭 다운 확인 후 즉시 진입
    else:
        skip
```

### loss_streak_cooldown 구현

```python
if loss_streak_cooldown is not None:
    consecutive_losses = count_recent_consecutive_losses(trades_so_far)
    if consecutive_losses >= loss_streak_cooldown:
        skip_next_signal = True  # 다음 1회 시그널 건너뜀
        # 건너뛴 후 카운터 리셋
```

### 시그널 파라미터 변경 지원

기존 시그널은 고정 파라미터로 구현되어 있으므로,
파라미터를 외부에서 주입할 수 있도록 확장:

```python
# 기존
def signal_S1_11(df):
    period = 7  # 고정
    ...

# 확장
def signal_S1_11(df, period=7):
    ...
```

**모든 시그널 함수에 파라미터를 kwargs로 받을 수 있게 수정.**

---

## 출력

### 파일 구조

```
results_v5/
├── grid_search/
│   ├── SEQ_S1.11_S4.04_is_grid.csv     # IS 그리드 결과
│   ├── S3.12_is_grid.csv
│   ├── S2.08_F3_is_grid.csv
│   ├── S1.11_is_grid.csv
│   └── SEQ_S1.11_S3.01_is_grid.csv
├── oos_validation/
│   ├── SEQ_S1.11_S4.04_oos.csv         # OOS 검증
│   └── ...
├── comparison_v5.csv                     # 원본 vs Phase4 최선 vs Phase5 최선
├── walk_forward_v5.csv
├── random_benchmark_v5.csv
├── june2026_spot_v5.csv
├── trade_analysis_v5.csv                # 개선 전후 보유기간별 WR 변화
└── dashboard_v5.html
```

### 대시보드 차트

| # | 차트 |
|---|------|
| 1 | **3단계 비교** — 원본 vs Phase4 vs Phase5 OOS return (전략별) |
| 2 | **파라미터 히트맵** — 전략별 SL×TP 조합의 IS 수익 히트맵 |
| 3 | **보유기간 WR 변화** — 원본 vs 최선의 보유기간별 WR 비교 (0일 문제 해결 확인) |
| 4 | **SL 히트율 변화** — 원본 vs 최선의 SL 발동 비율 |
| 5 | **Equity Curve** — 원본 vs Phase4 vs Phase5 누적 수익 |
| 6 | **자산별 성과 변화** — DOGE/XRP/BTC 제외 효과 (있다면) |
| 7 | **2026-06 스팟 체크** — 3버전 비교 |
| 8 | **WF 윈도우별 비교** — 원본 vs 최선의 5개 윈도우 수익 |

### 터미널 출력

```
══════════════════════════════════════════════════════════════
  Phase 5: Per-Strategy Parameter Tuning — Results
══════════════════════════════════════════════════════════════

📊 Grid search: 5 strategies × avg ~130 parameter sets

🏆 Strategy 1: SEQ_S1.11_S+S4.04_L
   v3 original:  OOS +45.0%  Sharpe 0.87  WF 5/5  p=.021
   v4 best(A4):  OOS +78.2%  Sharpe 0.78  WF 3/5  p=.005
   v5 best:      OOS +XX.X%  Sharpe X.XX  WF X/5  p=.XXX
   Key change:   [description]

🏆 Strategy 2: S3.12_S_aggr_F8
   v3 original:  OOS +43.6%  Sharpe 0.72  WF 3/5  p=.069
   v5 best:      OOS +XX.X%  Sharpe X.XX  WF X/5  p=.XXX
   Key change:   [description]
...

📁 Dashboard: results_v5/dashboard_v5.html
══════════════════════════════════════════════════════════════
```

---

## 핵심 원칙

1. **전략별 맞춤 최적화**: 동일 모듈을 전 전략에 일괄 적용(Phase 4)이 아니라, 각 전략의 고유한 약점에 맞춘 파라미터 탐색.
2. **단계적 그리드 축소**: 전수 탐색이 아닌 3~4단계 축소 탐색. 각 단계에서 상위 N개만 다음 단계로.
3. **IS 게이트 + OOS 검증**: IS에서 원본을 못 이기면 탈락. OOS에서 원본/Phase4를 못 이기면 탈락.
4. **과적합 경각심**: 특히 S2.08_F3(105건)은 파라미터 1개 바꿔도 과적합 위험. 작은 표본에서는 보수적 채택.
5. **비파괴적 확장**: 기존 v3/v4 코드를 수정하지 않고, 새 함수/파일로 확장.
6. **0일 보유 문제가 1순위**: 이것만 해결해도 전 전략에서 의미 있는 개선 기대.

---

## 실행

```bash
python main_v5.py                          # 전체 5개 전략 최적화
python main_v5.py --strategy SEQ_S1.11_S4  # 특정 전략만
python main_v5.py --stage 1                # IS 그리드만 (빠른 확인)
python main_v5.py --compare                # 3단계(v3/v4/v5) 비교표
```
