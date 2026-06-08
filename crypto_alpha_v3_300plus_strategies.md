# 크립토 알파 전략 발굴 v3 — 300+ 전략, 모듈형 설계

## 설계 철학

전략 = **Entry Signal** × **Direction** × **Exit Rule** × (optional) **Filter**

이 네 가지 빌딩 블록을 독립적으로 정의한 뒤 조합으로 300+ 전략을 생성한다.
모든 것은 OHLCV만으로 계산 가능해야 한다.

---

## 기존 인프라

Phase 1 프로젝트의 아래 코드를 재사용한다:
- `data/` — 9개 자산 OHLCV (BTC, ETH, SOL, DOGE, OP, AVAX, XRP, XLM, SUI)
- `src/data_loader.py` — 데이터 로드/전처리
- `src/indicators/_ta.py` — 기존 TA 직접 구현 (numpy 2.0 호환)
- `pandas_ta` 사용 금지. 모든 인디케이터는 pandas/numpy로 직접 구현.

---

# Part 1: Entry Signal Library (70+ 시그널)

모든 시그널은 아래 인터페이스를 따른다:

```python
def signal_XXX(df: pd.DataFrame, **params) -> pd.Series:
    """
    Args: OHLCV DataFrame (columns: open, high, low, close, volume)
    Returns: pd.Series[bool] — True인 날의 다음 날 시가에 진입
    """
```

## 그룹 S1: 프라이스 액션 / 캔들 패턴 (15개)

이것들은 인디케이터 없이 OHLC 자체에서 추출하는 패턴이다.
대부분 **LONG과 SHORT 양방향**으로 사용 가능.

| ID | 이름 | LONG 조건 | SHORT 조건 |
|----|------|----------|-----------|
| S1.01 | Bullish/Bearish Engulfing | 전일 음봉, 당일 양봉이 전일 몸통을 완전히 감쌈 (open<prev_close, close>prev_open) | 반대: 전일 양봉, 당일 음봉이 감쌈 |
| S1.02 | Hammer / Shooting Star | 하단 꼬리 ≥ 몸통의 2배, 상단 꼬리 < 몸통의 0.3배, 하락추세 중 (5일 ROC < -3%) | 상단 꼬리 ≥ 몸통 2배, 하단 꼬리 < 0.3배, 상승추세 중 |
| S1.03 | Doji + 방향 확인 | Doji (abs(close-open)/범위 < 0.1) 다음날 양봉 | Doji 다음날 음봉 |
| S1.04 | Inside Bar Breakout | 당일 High < 전일 High AND Low > 전일 Low → 다음날 전일 High 돌파 시 | 다음날 전일 Low 이탈 시 |
| S1.05 | Outside Bar (Key Reversal) | 당일 High > 전일 High AND Low < 전일 Low AND close > open, 하락추세 후 | 반대 조건, 상승추세 후 |
| S1.06 | Three White Soldiers / Black Crows | 3일 연속 양봉, 각각 전일 종가 위에서 시가, 종가 상승 | 3일 연속 음봉, 각각 전일 종가 아래에서 시가 |
| S1.07 | Morning Star / Evening Star | 3봉 패턴: 큰 음봉 → 작은 몸통(갭 다운) → 큰 양봉(1일차 50% 이상 회복) | 반대 |
| S1.08 | Pin Bar | 한쪽 꼬리 ≥ 전체 범위의 66%, 몸통은 범위의 상/하단 25% 이내 | 방향 반대 |
| S1.09 | Gap Fill | 갭 다운(open < prev_low) 발생 → 갭 메우기 기대 | 갭 업(open > prev_high) → 갭 메우기 기대 |
| S1.10 | N-Day Breakout (Donchian) | 종가 > N일 최고가 (N=20) | 종가 < N일 최저가 |
| S1.11 | Narrow Range (NR7) | 최근 7일 중 당일 범위(H-L)가 최소 → 다음날 방향 추종 | 같은 조건, 반대 방향 |
| S1.12 | 2-Bar Reversal | 2일간 큰 하락 후 (누적 < -4%), 둘째 날 종가가 범위 상단 25% | 2일 큰 상승 후 종가가 범위 하단 25% |
| S1.13 | 연속 N봉 반전 | N일 연속 음봉(N=3,4,5) 후 첫 양봉 | N일 연속 양봉 후 첫 음봉 |
| S1.14 | Range Contraction → Expansion | 3일 평균 범위 < 20일 평균 범위의 50% → 범위 2배 확장일에 방향 추종 | 같음, 방향 반대 |
| S1.15 | 종가 위치 반전 | 5일 연속 종가가 일일 범위 하위 25% → 반등 기대 | 5일 연속 상위 25% → 하락 기대 |

---

## 그룹 S2: 비주류/대안 모멘텀 인디케이터 (16개)

이것들은 RSI/MACD/Stochastic 같은 "교과서" 인디케이터의 **대안** 또는 **개선 버전**이다.

| ID | 이름 | 공식 요약 | LONG 시그널 | SHORT 시그널 |
|----|------|----------|------------|-------------|
| S2.01 | **Fisher Transform** | fisher = 0.5 × ln((1+x)/(1-x)), x = normalized(midprice, period) clipped to ±0.999 | Fisher 크로스업 (fisher > fisher_prev AND fisher_prev < -1) | Fisher 크로스다운 (fisher < fisher_prev AND fisher_prev > 1) |
| S2.02 | **Inverse Fisher Transform of RSI** | rsi_val = 0.1 × (RSI - 50), IFT = (e^(2×rsi_val) - 1) / (e^(2×rsi_val) + 1) | IFT 크로스업 -0.5 | IFT 크로스다운 +0.5 |
| S2.03 | **Connors RSI** | CRSI = (RSI(3) + StreakRSI(2) + PercentRank(ROC(1), 100)) / 3 | CRSI < 15 | CRSI > 85 |
| S2.04 | **TSI (True Strength Index)** | double-smoothed momentum: EMA(EMA(close-prev, r), s) / EMA(EMA(abs(close-prev), r), s) × 100, r=25, s=13, signal=EMA(TSI, 7) | TSI 크로스업 signal (TSI < 0 영역에서) | TSI 크로스다운 signal (TSI > 0에서) |
| S2.05 | **Coppock Curve** | WMA(14, ROC(14) + ROC(11)) | Coppock이 음수에서 양수로 전환 | Coppock이 양수에서 음수로 |
| S2.06 | **KST (Know Sure Thing)** | 4개 ROC의 가중합: w1×SMA(ROC(10),10) + w2×SMA(ROC(15),10) + w3×SMA(ROC(20),10) + w4×SMA(ROC(30),15), w=[1,2,3,4], signal=SMA(KST,9) | KST 크로스업 signal | KST 크로스다운 signal |
| S2.07 | **Aroon Oscillator** | AroonUp = ((period - days_since_highest) / period) × 100, AroonDown 유사, Osc = Up - Down | Osc > 50 크로스업 (or Osc 음→양) | Osc < -50 크로스다운 (or 양→음) |
| S2.08 | **Vortex Indicator** | +VM = abs(high - prev_low), -VM = abs(low - prev_high), +VI = sum(+VM,n)/sum(TR,n), -VI 유사 | +VI 크로스업 -VI | -VI 크로스업 +VI |
| S2.09 | **Elder Ray (Bull/Bear Power)** | BullPower = high - EMA(13), BearPower = low - EMA(13) | BearPower < 0에서 상승 전환 (전일 대비), EMA 상승 중 | BullPower > 0에서 하락 전환, EMA 하락 중 |
| S2.10 | **CMO (Chande Momentum Oscillator)** | CMO = (sumUp - sumDown) / (sumUp + sumDown) × 100, period=14 | CMO 크로스업 -50 | CMO 크로스다운 +50 |
| S2.11 | **DPO (Detrended Price Oscillator)** | DPO = close - SMA(20, shift=10+1) | DPO 음수에서 양수로 전환 | DPO 양수에서 음수로 |
| S2.12 | **Ultimate Oscillator** | UO = 100 × (4×avg7 + 2×avg14 + avg28) / 7, avg = sum(BP)/sum(TR) | UO < 30에서 bullish divergence | UO > 70에서 bearish divergence |
| S2.13 | **Stochastic RSI** | RSI를 Stochastic 공식에 대입: (RSI - min(RSI,n)) / (max(RSI,n) - min(RSI,n)) | StochRSI < 0.1에서 크로스업 0.2 | StochRSI > 0.9에서 크로스다운 0.8 |
| S2.14 | **RVI (Relative Vigor Index)** | RVI = SMA4(close-open) / SMA4(high-low), signal = SMA4(RVI) | RVI 크로스업 signal (음수 영역) | RVI 크로스다운 signal (양수 영역) |
| S2.15 | **Mass Index** | MI = sum(EMA(range,9)/EMA(EMA(range,9),9), 25). Squeeze = MI > 27 후 < 26.5 | Squeeze 발생 + 가격 하락추세 → 반전 기대 | Squeeze + 상승추세 → 반전 기대 |
| S2.16 | **Random Walk Index** | RWI_high = (H - L[n]) / (ATR × √n), RWI_low 유사, n=기간 | RWI_low > 1.0에서 하락 후 RWI_high 반등 | RWI_high > 1.0에서 상승 후 RWI_low 반등 |

---

## 그룹 S3: 대안 추세 인디케이터 (12개)

기존 SMA/EMA 크로스 대신, 노이즈에 더 강하거나 적응적인 추세 지표.

| ID | 이름 | 공식 요약 | LONG 시그널 | SHORT 시그널 |
|----|------|----------|------------|-------------|
| S3.01 | **Supertrend** | upperBand = hl2 + mult×ATR, lowerBand = hl2 - mult×ATR, 추세 반전 로직, mult=3, period=10 | 가격 > Supertrend 전환 (하락→상승) | 가격 < Supertrend 전환 |
| S3.02 | **Hull MA Cross** | HMA = WMA(2×WMA(n/2) - WMA(n), √n), n=20 | 가격 크로스업 HMA | 가격 크로스다운 HMA |
| S3.03 | **DEMA Cross** | DEMA = 2×EMA(n) - EMA(EMA(n)), fast=10, slow=30 | DEMA(10) 크로스업 DEMA(30) | 반대 |
| S3.04 | **TEMA Cross** | TEMA = 3×EMA - 3×EMA(EMA) + EMA(EMA(EMA)), fast=10, slow=30 | TEMA(10) 크로스업 TEMA(30) | 반대 |
| S3.05 | **KAMA (Kaufman Adaptive)** | ER = abs(direction)/volatility, SC = (ER×(fast-slow)+slow)², KAMA += SC×(price-KAMA) | 가격 크로스업 KAMA(10,2,30) | 반대 |
| S3.06 | **McGinley Dynamic** | MD = MD[-1] + (close - MD[-1]) / (N × (close/MD[-1])⁴) | 가격 크로스업 McGinley(14) | 반대 |
| S3.07 | **VIDYA** | VIDYA = α×CMO_ratio×close + (1-α×CMO_ratio)×VIDYA[-1] | 가격 크로스업 VIDYA | 반대 |
| S3.08 | **Linear Regression Slope** | N일 linear regression slope, signal=slope의 부호 전환 | slope 음→양 전환 | slope 양→음 전환 |
| S3.09 | **Linear Regression Channel 이탈** | 가격이 N일 회귀선 ± 2σ 채널 이탈 | 하단 이탈 후 회귀 (채널 내 복귀) | 상단 이탈 후 복귀 |
| S3.10 | **Heikin-Ashi 추세 전환** | HA_close = (O+H+L+C)/4, HA_open = (prev_HA_O+prev_HA_C)/2 | 3+봉 HA 음봉 후 HA 양봉 전환 | 반대 |
| S3.11 | **Ichimoku TK Cross (단독)** | Tenkan(9) vs Kijun(26) 크로스 | Tenkan 크로스업 Kijun (구름 위) | 반대 (구름 아래) |
| S3.12 | **Pivot Point Breakout** | PP = (H+L+C)/3, R1 = 2PP-L, S1 = 2PP-H (전일 기준) | 종가 > R1 돌파 | 종가 < S1 이탈 |

---

## 그룹 S4: 변동성/레인지 기반 (12개)

| ID | 이름 | 공식 요약 | LONG 시그널 | SHORT 시그널 |
|----|------|----------|------------|-------------|
| S4.01 | **Squeeze Momentum (TTM 개념)** | BB(20,2) 안에 Keltner(20,1.5) 들어옴 = squeeze on. Squeeze 해제 시 momentum 방향 추종. momentum = close - avg(highest(high,20), lowest(low,20))/2의 linear regression value | Squeeze 해제 + momentum 양수 | Squeeze 해제 + momentum 음수 |
| S4.02 | **Chaikin Volatility** | CV = EMA(high-low, 10)의 ROC(10) × 100 | CV 극단 상승(>50) 후 하락 전환 + 가격 저점 | CV 극단 후 하락 + 가격 고점 |
| S4.03 | **Historical Vol Percentile** | 20일 수익률 std의 252일 percentile rank | HV pctile < 10% (극저변동성) → 브레이크아웃 방향 추종 | 같은 조건, 방향 반대 |
| S4.04 | **Normalized ATR** | NATR = ATR(14) / close × 100 | NATR > 90th pctile(60일) + 하락일 → 과잉반응 반전 | NATR 급등 + 상승 추세 끝 |
| S4.05 | **Yang-Zhang Volatility** | O-C, C-O variance 조합 (가장 효율적 OHLC vol estimator) | YZ vol의 급등 (2σ 초과, 60일 기준) + 하락일 → 반전 | 반대 |
| S4.06 | **Parkinson Volatility Ratio** | Parkinson/close-close vol ratio가 극단적 → 일중 변동 과다 | ratio > 2.0 + 하락 → 과매도 반등 | ratio > 2.0 + 상승 → 과열 |
| S4.07 | **BB %B 극단** | %B = (close - lower) / (upper - lower) | %B < 0 (하단 이탈) | %B > 1 (상단 이탈) |
| S4.08 | **Keltner-BB Spread** | BB width - Keltner width | spread 음→양 전환 (squeeze 해제) + 방향 | 반대 방향 |
| S4.09 | **ATR Trailing Stop 반전** | close ± mult × ATR(14), mult=3, trailing 로직 | trailing stop 상향 반전 (하락→상승) | 하향 반전 |
| S4.10 | **Chandelier Exit 반전** | 22일 최고/최저 - 3×ATR(22) | 하향 chandelier 돌파 후 회복 | 상향 chandelier 이탈 |
| S4.11 | **변동성 수축 패턴 (VCP)** | 연속적으로 줄어드는 범위 (3회+), 마지막 수축의 상단 돌파 | 상단 돌파 | 하단 이탈 |
| S4.12 | **Ulcer Index 기반** | UI = √(mean(drawdown²)), period=14 | UI > 90th pctile → 과매도 반등 | UI < 10th pctile에서 상승 후 UI 급등 시작 |

---

## 그룹 S5: 대안 거래량 인디케이터 (10개)

| ID | 이름 | 공식 요약 | LONG 시그널 | SHORT 시그널 |
|----|------|----------|------------|-------------|
| S5.01 | **MFI (Money Flow Index)** | volume-weighted RSI: TP=(H+L+C)/3, MF=TP×V, MFI=100-100/(1+posMF/negMF), period=14 | MFI < 20 (과매도) | MFI > 80 (과매수) |
| S5.02 | **Ease of Movement** | EMV = ((H+L)/2 - (prev_H+prev_L)/2) / (V / (H-L)), SMA(EMV, 14) | EMV SMA 음→양 전환 | 양→음 전환 |
| S5.03 | **Klinger Volume Oscillator** | KVO = EMA(34, VF) - EMA(55, VF), VF = V × sign(trend) × abs(dm/cm), signal = EMA(13, KVO) | KVO 크로스업 signal (음수 영역) | 크로스다운 (양수 영역) |
| S5.04 | **Negative Volume Index** | NVI: 거래량 감소일에만 수익률 누적. Signal = EMA(NVI, 255) | NVI > signal (스마트머니 매집) | NVI < signal |
| S5.05 | **Positive Volume Index** | PVI: 거래량 증가일에만 수익률 누적. Signal = EMA(PVI, 255) | PVI 크로스다운 signal (군중 이탈 → 역행) | PVI 크로스업 signal (군중 추종) |
| S5.06 | **Volume Oscillator** | VO = (EMA(V,5) - EMA(V,20)) / EMA(V,20) × 100 | VO 음수 극단 (< -30%) + 가격 하락 → 셀링 소진 | VO 양수 극단 (> 50%) + 가격 상승 → 클라이맥스 |
| S5.07 | **VWAP 괴리율** | VWAP = cumsum(TP×V) / cumsum(V) (rolling 20일), deviation = (close - VWAP) / VWAP | 괴리율 < -2σ (20일 기준) | 괴리율 > +2σ |
| S5.08 | **A/D Oscillator** | AD = ((C-L)-(H-C))/(H-L)×V 누적, Osc = EMA(AD,3) - EMA(AD,10) | Osc 음→양 전환 + 가격 하락 중 (다이버전스) | 양→음 + 가격 상승 중 |
| S5.09 | **Volume Price Confirmation** | 가격 N일 방향 vs 거래량 N일 추세 불일치 | 가격 하락 + 거래량 감소 (N=5) → 매도 소진 | 가격 상승 + 거래량 감소 → 매수 소진 |
| S5.10 | **Relative Volume (RVOL)** | RVOL = today_volume / SMA(volume, 20) | RVOL > 3 + 양봉 (대량 매수) | RVOL > 3 + 음봉 (대량 매도) |

---

## 그룹 S6: 통계/수학적 시그널 (15개)

교과서 TA가 아닌, 통계학/시계열 분석에서 차용한 신호.

| ID | 이름 | 공식 요약 | LONG 시그널 | SHORT 시그널 |
|----|------|----------|------------|-------------|
| S6.01 | **Return Z-Score** | z = (ret_N - mean(ret_N, lookback)) / std(ret_N, lookback), N=1, lookback=60 | z < -2.0 (극단 하락, 평균회귀) | z > 2.0 (극단 상승) |
| S6.02 | **Cumulative Return Z-Score** | z of 5-day cumulative return, lookback=120 | z < -2.0 | z > 2.0 |
| S6.03 | **Price Percentile Rank** | 현재 종가의 252일 내 percentile | pctile < 10% | pctile > 90% |
| S6.04 | **Hurst Exponent (간이)** | R/S analysis 또는 variance ratio로 간이 추정, period=60 | Hurst < 0.4 (mean-reverting) + 가격 하락 | Hurst > 0.6 (trending) + 하락 시작 |
| S6.05 | **Autocorrelation Flip** | lag-1 autocorrelation of returns, rolling 20일 | AC < -0.3 (과잉 반전, 곧 추세 전환) | AC > 0.3 (추세 지속 중, 추종) |
| S6.06 | **Skewness Shift** | rolling 20일 수익률 skewness | skew < -1.5 (좌편향 극단 → 반등) | skew > 1.5 (우편향 → 하락) |
| S6.07 | **Kurtosis Spike** | rolling 20일 수익률 kurtosis | kurtosis > 6 (fat tail 출현) + 하락일 → 변동성 소진 반등 | kurtosis > 6 + 상승 후 → 반전 |
| S6.08 | **Entropy of Returns** | Shannon entropy of binned return distribution, rolling 30일 | entropy 급락 (< 20th pctile) → 시장 편향 극단 → 반전 | 같은 조건 |
| S6.09 | **Variance Ratio** | VR = var(ret_5d) / (5 × var(ret_1d)), rolling 60일 | VR < 0.7 (과잉 반전, mean-revert) + 하락 | VR > 1.3 (과잉 추세) + 추세 추종 |
| S6.10 | **Distance from MA (Detrended)** | (close - SMA(50)) / SMA(50) × 100 | distance < -15% (과잉 이탈, 평균회귀) | distance > +15% |
| S6.11 | **Consecutive Directional Days** | 연속 상승/하락 일수 카운트 | 5일+ 연속 하락 → 반전 기대 | 5일+ 연속 상승 → 반전 기대 |
| S6.12 | **High-Low Range Ratio** | today_range / avg_range(20) | ratio > 2.5 + 하락일 → 과잉 공포 반등 | ratio > 2.5 + 상승일 → 과열 |
| S6.13 | **Close Location Value (CLV)** | CLV = (2C - H - L) / (H - L), rolling 5일 평균 | CLV_avg < -0.7 (계속 저점 마감 → 소진) | CLV_avg > 0.7 |
| S6.14 | **Median Reversion** | (close - rolling_median(50)) / rolling_mad(50) | < -2.5 (median 대비 극단 저평가) | > +2.5 |
| S6.15 | **Return Regime (Hidden State)** | 20일 수익률의 부호 전환 빈도: 최근 10일 중 부호 전환 횟수 | 전환 > 7 (choppy → 방향 결정 임박) + 마지막 양봉 | + 마지막 음봉 |

---

## 그룹 S7: 크로스에셋 / 상대강도 (12개)

이 시그널은 **단일 자산이 아닌 전체 유니버스를 참조**한다.

| ID | 이름 | 조건 | 방향 |
|----|------|------|------|
| S7.01 | **BTC 대비 상대강도 반전 (약세→강세)** | ALT/BTC ratio의 RSI(14) < 30에서 반등 | LONG ALT |
| S7.02 | **BTC 대비 상대강도 반전 (강세→약세)** | ALT/BTC ratio의 RSI(14) > 70에서 하락 | SHORT ALT |
| S7.03 | **Breadth Thrust** | EMA(20) 위 자산 비율이 < 20%에서 > 50%로 급등 (3일 내) | LONG (전 자산) |
| S7.04 | **Breadth Collapse** | EMA(20) 위 비율 > 80%에서 < 50%로 급락 | SHORT (전 자산) |
| S7.05 | **Pair Spread Mean Reversion** | ETH/BTC, SOL/BTC 등 spread의 z-score(30일) < -2 | LONG 분자 (or SHORT 분모) |
| S7.06 | **Pair Spread Momentum** | spread z-score > +2 지속 중 (추세 추종) | LONG 분자 |
| S7.07 | **Sector Rotation: L1 vs Meme** | L1 섹터(BTC,ETH,SOL,AVAX,SUI) 7일 수익률 > Meme(DOGE) + Infra(OP,XRP,XLM) | LONG L1, SHORT 나머지 |
| S7.08 | **Correlation Breakdown** | 특정 자산과 BTC의 30일 상관관계 < 0.3 (디커플링) | 독립 움직임 → 해당 자산 추세 추종 |
| S7.09 | **Lead-Lag: BTC 선행** | BTC 2일 전 수익률 vs ALT 당일 수익률의 상관이 높을 때 (>0.5), BTC가 2일 전 양봉 | LONG ALT |
| S7.10 | **Market Cap 가중 모멘텀** | 전체 자산 volume-weighted 7일 수익률 < -10% | LONG (전체 패닉 반전) |
| S7.11 | **Relative Strength Ranking** | 9개 자산 중 20일 수익률 순위 바닥(8~9위) → 반등 기대 | LONG |
| S7.12 | **상관관계 극단 + 약세** | 30일 전 자산 상관관계 > 0.9 (동조화) + 전체 하락 → 가장 덜 빠진 자산 | LONG 상대 강세 자산 |

---

## 그룹 S8: 기존 인디케이터 재활용 — 비표준 사용법 (10개)

"교과서적" 인디케이터를 **비표준적으로** 사용하는 시그널.

| ID | 이름 | 비표준 사용법 | LONG | SHORT |
|----|------|-------------|------|-------|
| S8.01 | **RSI of Volume** | RSI를 가격이 아닌 거래량에 적용 (period=14) | VolRSI < 20 → 거래량 소진, 가격 반등 기대 | VolRSI > 80 → 거래량 클라이맥스 |
| S8.02 | **MACD of ATR** | MACD를 ATR 시계열에 적용 | ATR-MACD 데드크로스 → 변동성 감소 시작, 저점 탈출 | ATR-MACD 골든크로스 → 변동성 급증 시작 |
| S8.03 | **BB of RSI** | RSI에 Bollinger Band 적용 (RSI의 과매도를 동적으로 정의) | RSI < RSI_BB_lower → 동적 과매도 | RSI > RSI_BB_upper → 동적 과매수 |
| S8.04 | **OBV의 MA Cross** | OBV에 EMA(10)/EMA(30) 크로스 적용 | OBV EMA 골든크로스 | OBV EMA 데드크로스 |
| S8.05 | **ATR Ratio (short/long)** | ATR(5) / ATR(20) — 단기 변동성 / 장기 변동성 | ratio > 1.5에서 하락 후 ratio < 0.8로 안정 | ratio > 1.5 + 상승 추세 |
| S8.06 | **이중 타임프레임 RSI** | RSI_weekly (5일 대체) < 30 AND RSI_daily(14) 30 크로스업 | 양쪽 과매도 확인 후 반등 | 양쪽 과매수 확인 후 하락 |
| S8.07 | **MACD 히스토그램 다이버전스** | 가격 lower low + MACD hist higher low | LONG (히든 불리시 다이버전스) | 가격 higher high + hist lower high |
| S8.08 | **Volume-Weighted RSI** | RSI 계산 시 각 bar의 price change를 volume으로 가중 | VW-RSI < 25 | VW-RSI > 75 |
| S8.09 | **Stochastic of OBV** | OBV에 Stochastic 공식 적용 (20일 lookback) | StochOBV < 10 → 매집 저점 | StochOBV > 90 → 분배 고점 |
| S8.10 | **RSI 변화 속도** | ROC of RSI(14), period=5 | RSI_ROC < -30 (RSI 급락 → 과잉반응) + RSI < 40 | RSI_ROC > 30 (RSI 급등) + RSI > 60 |

---

# Part 2: Exit Rule Library (12가지)

모든 전략은 아래 exit rule 중 하나(또는 조합)를 사용한다.

| ID | 이름 | 로직 |
|----|------|------|
| E1 | **고정 기간** | N일 후 청산 (N = 3, 5, 7, 10, 14, 21) |
| E2 | **반대 시그널** | 진입과 반대 방향 시그널 발생 시 청산 |
| E3 | **Stop Loss** | 진입가 대비 -X% 도달 (보유 중 Low/High 체크) |
| E4 | **Take Profit** | 진입가 대비 +Y% 도달 |
| E5 | **Trailing Stop** | 보유 중 최고점(LONG)/최저점(SHORT) 대비 -Z% 이탈 |
| E6 | **인디케이터 중립 복귀** | RSI 50 복귀, BB 중간선 복귀, 등 |
| E7 | **시간 기반 동적** | max(7, 2×ATR_period) — 변동성에 비례한 보유 기간 |
| E8 | **ATR 기반 Stop** | 진입가 ± N×ATR(14) (N=2,3) |
| E9 | **Chandelier Exit** | 보유 중 최고/최저 ± 3×ATR(22) |
| E10 | **Profit Target + Trailing** | +3% 도달 시 trailing stop 활성화 (1.5×ATR) |
| E11 | **Break-Even Stop** | +2% 도달 후 진입가로 stop 이동 |
| E12 | **First Profitable Close** | 종가 기준 첫 수익 발생일에 청산 (빠른 확정) |

---

# Part 3: Filter Library (10가지)

필터는 entry signal에 **추가 조건**으로 적용되어 시그널을 걸러낸다.
필터가 True일 때만 해당 시그널이 활성화된다.

| ID | 이름 | True 조건 | 용도 |
|----|------|----------|------|
| F1 | **상승 추세 필터** | close > EMA(50) | LONG 전용 필터 (추세 방향 매수) |
| F2 | **하락 추세 필터** | close < EMA(50) | SHORT 전용 필터 |
| F3 | **횡보 필터** | ADX(14) < 20 | 역추세(mean-reversion) 전략에 |
| F4 | **추세 필터** | ADX(14) > 25 | 추세 추종 전략에 |
| F5 | **변동성 고조 필터** | ATR(14) > SMA(ATR, 60) × 1.5 | 변동성 전략에 |
| F6 | **변동성 저조 필터** | ATR(14) < SMA(ATR, 60) × 0.7 | 브레이크아웃 대기 |
| F7 | **거래량 확인 필터** | volume > SMA(volume, 20) × 1.5 | 유의미한 거래량 동반 시에만 |
| F8 | **Market Breadth 필터** | EMA(20) 위 자산 비율 < 40% | 시장 전체 약세 시에만 |
| F9 | **BTC 방향 필터** | BTC의 5일 수익률 > 0 (or < 0) | BTC 추세 방향에 맞춰 |
| F10 | **Regime 필터** | 60일 실현변동성 percentile > 70% | 고변동성 국면에서만 |

---

# Part 4: 전략 생성 — 300+ 자동 조합

## 4.1 생성 규칙

### Layer 1: 기본 단일 전략 (시그널 × 방향 × 기본 Exit)

```python
base_strategies = []
for signal in ALL_SIGNALS:  # 70+ 시그널
    for direction in signal.available_directions:  # LONG, SHORT, or both
        # 기본 exit: E3(-5% SL) + E4(+10% TP) + E1(10일 max hold)
        base_strategies.append(Strategy(signal, direction, exit=[E3, E4, E1]))
```

→ 약 **130개** 기본 전략 (70 시그널 × ~1.85 avg directions)

### Layer 2: Exit 변형 (상위 50개 기본 전략에 적용)

상위 50개 시그널에 대해 exit rule 변형:

| 변형 | Exit 조합 | 개수 |
|------|----------|------|
| Conservative | E3(-3%) + E4(+5%) + E1(7일) | ×50 |
| Aggressive | E3(-7%) + E4(+15%) + E1(14일) | ×50 |
| Trailing | E5(trailing -3%) + E1(14일) | ×50 |
| Quick Scalp | E12(first profitable close) + E3(-3%) + E1(5일) | ×50 |
| ATR-based | E8(2×ATR SL) + E8(3×ATR TP) + E1(10일) | ×50 |

→ 50 × 5 = **250개** (기본 50과 중복 제외 시 ~200개 추가)

**Layer 1 + Layer 2 ≈ 330개**

### Layer 3: 필터 적용 (상위 30개 전략에)

IS 테스트 후 상위 30개 전략에 10개 필터 적용:
→ 30 × 10 = **300개** 추가 (하지만 이것은 Phase 4 이후 실행)

### Layer 4: AND 조합 (Phase 4 이후)

IS 상위 20개 전략에서 2개씩 AND 조합:
→ C(20,2) = 190개 × 방향 = **190개**

### Layer 5: SEQUENTIAL 조합 (Phase 4 이후)

IS 상위 SHORT 5개 → 상위 LONG 5개 전환:
→ 5 × 5 = **25개**

### 총 전략 수

| Layer | 전략 수 | 누적 |
|-------|--------|------|
| L1: 기본 | ~130 | 130 |
| L2: Exit 변형 | ~200 | 330 |
| L3: 필터 (post-IS) | ~300 | 630 |
| L4: AND 조합 (post-IS) | ~190 | 820 |
| L5: SEQUENTIAL (post-IS) | ~25 | 845 |

**Phase 3(IS 백테스트) 시점에서 ~330개, 전체적으로 ~845개 후보.**
Phase 4 이후 조합은 IS 상위에서만 생성하므로 과적합 리스크 관리.

---

## 4.2 구현: Strategy Generator

```python
class StrategyGenerator:
    """전략을 프로그래밍적으로 대량 생성"""

    def generate_layer1(self) -> list[Strategy]:
        """70+ 시그널 × 방향 × 기본 exit → ~130개"""

    def generate_layer2(self, top_n: int = 50) -> list[Strategy]:
        """상위 시그널에 exit 변형 적용 → ~200개 추가"""

    def generate_layer3(self, top_strategies: list, filters: list) -> list[Strategy]:
        """IS 상위 전략에 필터 적용 → ~300개"""

    def generate_layer4(self, top_strategies: list, k: int = 2) -> list[Strategy]:
        """IS 상위에서 AND 조합 → ~190개"""

    def generate_layer5(self, top_short: list, top_long: list) -> list[Strategy]:
        """SHORT→LONG 전환 → ~25개"""
```

---

## 4.3 전략 네이밍 규칙

```
{Signal_ID}_{Direction}_{Exit_Variant}_{Filter}
```

예시:
- `S2.03_L_base` → Connors RSI, LONG, 기본 exit
- `S2.03_L_trailing` → Connors RSI, LONG, trailing exit
- `S2.03_L_trailing_F1` → + 상승추세 필터
- `S1.01_S_quick+S6.01_S_quick` → AND 조합

---

# Part 5: 백테스트 엔진

## 5.1 트레이드 시뮬레이션 (v2에서 그대로 유지)

```python
@dataclass
class Trade:
    strategy_id: str
    symbol: str
    direction: str          # "LONG" or "SHORT"
    entry_date: date
    entry_price: float      # 시그널 다음 날 시가(Open)
    exit_date: date
    exit_price: float
    exit_reason: str        # "signal", "stop_loss", "take_profit",
                            # "trailing_stop", "max_hold", "atr_stop",
                            # "first_profit", "break_even"
    holding_days: int
    pnl_pct: float          # 수익률 (슬리피지 차감 후)
```

## 5.2 진입/청산 가격 규칙

- **진입**: 시그널 발생일의 **다음 날 시가(Open)**
- **청산**: 청산 조건 충족일의 **당일 종가(Close)**
- **SL/TP 체크**: 보유 중 High/Low로 intraday 발동 확인
  - LONG SL: 해당일 Low ≤ entry × (1 + sl_pct) → 발동가 = entry × (1 + sl_pct)
  - LONG TP: 해당일 High ≥ entry × (1 + tp_pct)
  - SHORT: 반대
  - 같은 날 SL+TP 모두 가능 → **SL 우선** (보수적)
- **슬리피지**: 편도 0.1% (왕복 0.2%)
- **Non-overlapping**: 포지션 보유 중 신규 시그널 무시
- **Cooldown**: 청산 후 같은 자산 3일 재진입 금지

## 5.3 벤치마크

모든 수익은 아래 대비 **초과수익(알파)**도 함께 보고:
- **Buy & Hold BTC** (같은 기간)
- **Buy & Hold 해당 자산** (자산별 비교 시)
- **Random Entry** (같은 빈도의 랜덤 진입, 1000회 시뮬레이션 평균)

---

# Part 6: 평가 메트릭

## 6.1 전략당 × 자산당

| 메트릭 | 정의 |
|--------|------|
| Total Return (%) | Σ(trade pnl), 비복리 |
| # Trades | 총 트레이드 수 (최소 10건 필수) |
| Win Rate | 수익 trades / 전체 |
| Avg Win | 수익 trades 평균 수익률 |
| Avg Loss | 손실 trades 평균 손실률 |
| Profit Factor | Σ(wins) / Σ(abs(losses)) |
| Expectancy | WR × AvgWin - (1-WR) × abs(AvgLoss) |
| Sharpe-like | mean(trade_rets) / std(trade_rets) × √(252/avg_hold) |
| Max Consec. Losses | 최대 연속 손실 |
| Max Single Loss | 단일 최대 손실 |
| Avg Holding Days | 평균 보유 기간 |
| Alpha vs B&H | Total Return - Buy&Hold Return (같은 기간) |

## 6.2 크로스에셋 통합

| 메트릭 | 정의 |
|--------|------|
| Cross-Asset Avg Return | 9개 자산 평균 |
| Positive Asset Count | Return > 0인 자산 수 (/9) |
| Cross-Asset Sharpe | 평균 Sharpe |
| Consistency | Profit Factor > 1.0인 자산 비율 |
| Worst Case | 최악 자산 Return |
| Median Return | 9개 중 중간값 (outlier 강건) |

## 6.3 종합 랭킹

```python
composite = (
    0.25 * norm(cross_asset_median_return) +  # 중간값 사용 (outlier 방어)
    0.20 * norm(cross_asset_sharpe) +
    0.20 * norm(consistency) +
    0.15 * norm(avg_profit_factor) +
    0.10 * norm(avg_expectancy) +
    0.10 * norm(positive_asset_count / 9)
)
```

---

# Part 7: OOS 검증 & 과적합 필터

## 7.1 시간 분할

| 구간 | 기간 | 용도 |
|------|------|------|
| In-Sample | 2022-01 ~ 2024-06 | 전략 선택, exit/filter 최적화 |
| Out-of-Sample | 2024-07 ~ 2026-06 | 최종 검증 |

## 7.2 절차

1. Layer 1+2 전략 (~330개) → IS 백테스트 → IS 랭킹
2. IS 상위 50개 → Layer 3 필터 적용 (×10) → IS 재테스트 → IS 상위 갱신
3. IS 상위 30개 → Layer 4/5 조합 → IS 테스트 → IS 최종 상위 30개
4. IS 최종 상위 30개 → **OOS 백테스트** → OOS 랭킹
5. OOS 생존 기준:
   - OOS Total Return > 0 → "생존"
   - OOS Profit Factor > 1.0 → "유효"
   - OOS Sharpe > IS Sharpe × 0.5 → "견고"

## 7.3 Walk-Forward (5 윈도우)

| Window | IS | OOS |
|--------|-----|-----|
| 1 | 2022-01 ~ 2023-12 | 2024-01 ~ 2024-06 |
| 2 | 2022-07 ~ 2024-06 | 2024-07 ~ 2024-12 |
| 3 | 2023-01 ~ 2024-12 | 2025-01 ~ 2025-06 |
| 4 | 2023-07 ~ 2025-06 | 2025-07 ~ 2025-12 |
| 5 | 2024-01 ~ 2025-12 | 2026-01 ~ 2026-06 |

**3/5+ OOS 양수 = Walk-Forward Validated.**

## 7.4 Random Entry Benchmark

**과적합 최종 방어선.** OOS 상위 전략의 실제 수익이 랜덤 진입 대비 통계적으로 유의한지 확인:
- 같은 트레이드 빈도로 랜덤 날짜에 진입 × 1000회 시뮬레이션
- 전략 수익 > 랜덤 수익의 95th percentile → p < 0.05

## 7.5 파라미터 민감도

OOS 상위 10개의 핵심 파라미터를 ±30% 변동:
- 인디케이터 period: ×0.7, ×0.85, ×1.0, ×1.15, ×1.3
- SL: [-3%, -5%, -7%, -10%]
- TP: [+5%, +7%, +10%, +15%]
- Max hold: [5, 7, 10, 14, 21일]

**성능 변동 ±30% 이내 = 파라미터 안정적.**

---

# Part 8: 2026년 6월 스팟 체크

OOS 검증 통과 전략의 **2026-05-25 ~ 2026-06-06** 구간 상세:

- 시그널 발생 날짜 & 자산
- 진입가, 현재가(또는 청산가), P&L
- 방향 (LONG/SHORT)
- 아직 보유 중이면 MtM 표시

---

# Part 9: 출력

## 9.1 파일 구조

```
project/
├── data/                              # (기존)
├── src/
│   ├── data_loader.py                 # (기존)
│   ├── indicators/                    # (기존 + 확장)
│   │   ├── _ta.py                     # 기존 TA 구현
│   │   ├── _ta_extended.py            # 신규 인디케이터 (S2~S6)
│   │   └── _patterns.py              # 캔들 패턴 (S1)
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── registry.py               # 전체 시그널 등록/검색
│   │   ├── price_action.py            # S1
│   │   ├── alt_momentum.py            # S2
│   │   ├── alt_trend.py               # S3
│   │   ├── volatility.py              # S4
│   │   ├── alt_volume.py              # S5
│   │   ├── statistical.py             # S6
│   │   ├── cross_asset.py             # S7
│   │   └── nonstandard.py             # S8
│   ├── exits.py                       # E1~E12
│   ├── filters.py                     # F1~F10
│   ├── strategy.py                    # Strategy 클래스 + generator
│   ├── backtester_v3.py               # P&L 백테스트 엔진
│   ├── scorer_v3.py                   # 메트릭 & 랭킹
│   ├── oos_validator.py               # IS/OOS + Walk-Forward
│   ├── random_benchmark.py            # 랜덤 진입 벤치마크
│   ├── sensitivity.py                 # 파라미터 민감도
│   ├── spot_check.py                  # 6월 스팟 체크
│   └── dashboard_v3.py                # 시각화
├── results_v3/
│   ├── all_trades.csv
│   ├── strategy_scores_is.csv
│   ├── strategy_scores_oos.csv
│   ├── walk_forward.csv
│   ├── random_benchmark.csv
│   ├── sensitivity.csv
│   ├── june2026_spot.csv
│   └── dashboard.html
├── main_v3.py
└── requirements.txt
```

## 9.2 대시보드 차트 (12개)

| # | 차트 |
|---|------|
| 1 | **전략 IS 랭킹** — 종합 점수 내림차순, 카테고리별 색, 상위 30개 |
| 2 | **IS vs OOS 산점도** — x=IS return, y=OOS return, 대각선=일관, 상위 30개 |
| 3 | **Return vs Sharpe 산점도** — 크기=트레이드 수, 색=카테고리 |
| 4 | **Equity Curve** — 상위 5개 전략 누적 수익률 + BTC B&H 벤치마크 |
| 5 | **자산×전략 히트맵** — 상위 20 전략 × 9 자산, 색=수익률 |
| 6 | **트레이드 수익률 분포** — 상위 5개 전략의 히스토그램 (vs 랜덤 진입 분포) |
| 7 | **Win Rate vs Profit Factor** — 산점도, 우상단=좋은 전략 |
| 8 | **월별 수익 히트맵** — 상위 5개, x=월, y=연도, 색=수익 |
| 9 | **Walk-Forward 결과** — 5개 윈도우 각각의 OOS 수익, 전략별 |
| 10 | **파라미터 민감도** — 상위 3개 전략, 파라미터 vs 성과 |
| 11 | **시그널 그룹별 평균 성과** — S1~S8 그룹별 평균 return/sharpe |
| 12 | **2026-06 타임라인** — BTC 가격 + 상위 전략 진입/청산 마커 |

## 9.3 터미널 출력

```
══════════════════════════════════════════════════════════════
  Crypto Alpha Strategy Discovery v3 — Results
══════════════════════════════════════════════════════════════

📊 Universe: 9 assets × 2022-01 ~ 2026-06
🔄 Strategies evaluated: XXX (Layer1: 130, Layer2: 200, L3+: XXX)
📈 Total trades simulated: XX,XXX

🏆 IS Top 10 Strategies:
 #1  [S2.03_L_trailing] Connors RSI LONG, Trailing Exit
     Return: +XXX%, Sharpe: X.XX, WR: XX%, PF: X.X, Trades: XXX
 ...

✅ OOS Validated (Top 10 → OOS survivors):
 #1  [S2.03_L_trailing] OOS Return: +XX%, Sharpe: X.XX ✓ Robust
 ...
 Survival rate: X/10, Walk-Forward validated: X/10

📊 vs Random Entry:
 [S2.03_L_trailing] Strategy: +XX% vs Random 95th pctile: +XX% → p < 0.05 ✓

📅 June 2026 Spot Check:
 [S2.03_L_trailing] BTC: LONG 06-04 @ $XX,XXX → +X.X%
 ...

🏷️ Best by Category:
 Dip Buy:    [XXX] +XX%
 Momentum:   [XXX] +XX%
 Volatility: [XXX] +XX%
 Alt Momentum: [XXX] +XX%
 Statistical: [XXX] +XX%
 Cross-Asset: [XXX] +XX%

📁 Dashboard: results_v3/dashboard.html
══════════════════════════════════════════════════════════════
```

---

# Part 10: 실행

```bash
# 전체 파이프라인
python main_v3.py

# 빠른 테스트 (Layer1만, OOS/WF 생략)
python main_v3.py --quick

# Layer1+2까지만 (필터/조합 전)
python main_v3.py --layers 1,2

# 6월 스팟 체크만
python main_v3.py --spot-only
```

---

# Part 11: 핵심 원칙

1. **Look-ahead bias 금지**: 시그널=종가 기준, 진입=다음날 시가. _ta_extended.py의 모든 함수는 backward-looking only.
2. **슬리피지 포함**: 편도 0.1%. 이것 없이 산출된 수익은 환상.
3. **Non-overlapping**: 한 자산에 동시 포지션 불가.
4. **최소 트레이드 수**: 10건 미만은 통계적 무의미 → 결과에서 제외.
5. **과적합 방어**: IS/OOS 분리, Walk-Forward, Random Benchmark 3중 검증.
6. **비복리 합산**: 개별 트레이드 수익의 단순 합.
7. **벤치마크 대비**: 절대 수익만으로 판단하지 않음. B&H, 랜덤 대비 알파 필수.
8. **실행 속도**: 330+ 전략 × 9 자산 = 3000+ 백테스트. 벡터화 필수, tqdm으로 진행 상황 표시.
9. **중간 저장**: 각 Phase 완료 시 CSV 저장 → 재실행 시 캐싱.
10. **SUI/OP 주의**: 상장일 이전 데이터 없음. 트레이드 수 < 10이면 해당 자산 결과 제외.
