export const meta = {
  name: 'v3-engine-audit',
  description: 'Adversarial correctness + look-ahead audit of the v3 engine and analysis modules',
  phases: [
    { title: 'Review', detail: 'one reviewer per dimension hunts for defects' },
    { title: 'Verify', detail: 'independent skeptic confirms or refutes each finding' },
  ],
}

const SRC = 'C:\\\\AI_AGENTS\\\\NewStrategy\\\\src'
const SPEC = 'C:\\\\AI_AGENTS\\\\NewStrategy\\\\crypto_alpha_v3_300plus_strategies.md'

const DIMS = [
  {
    key: 'backtester',
    files: `${SRC}\\\\backtester_v3.py`,
    focus: `P&L trade simulator. Verify against spec section 5.2:
- entry is the bar AFTER a signal, filled at that bar's OPEN (e=i+1); NO look-ahead.
- exit at condition bar CLOSE except intraday SL/TP/trailing/ATR/chandelier at trigger level via High/Low.
- same-bar SL beats TP (conservative). Multiple stops collapse to the TIGHTEST (max for LONG, min for SHORT).
- trailing stop must trail the peak/trough through the PRIOR bar only (no within-bar peek where today's high sets a stop today's low then hits).
- slippage 0.1% per side applied to BOTH entry and exit fills, correct sign for LONG vs SHORT.
- SHORT pnl sign = (entry_fill - exit_fill)/entry_fill; LONG = (exit_fill - entry_fill)/entry_fill.
- non-overlapping per asset + 3-bar cooldown (blocked_until = exit_index + cooldown).
- date_mask restricts ENTRY dates only.
- ATR-at-entry used for ATR stops (atr_arr[e]); chandelier uses prior-bar level.
Hunt for: off-by-one in entry/exit indexing, wrong slippage sign, trailing within-bar look-ahead, max_hold counting error, cooldown letting overlapping trades through.`,
  },
  {
    key: 'scorer',
    files: `${SRC}\\\\scorer_v3.py`,
    focus: `Metrics + cross-asset aggregation + composite. Verify against spec section 6:
- profit_factor = sum(wins)/sum(abs(losses)); expectancy = WR*AvgWin-(1-WR)*abs(AvgLoss).
- Sharpe-like = mean(trade_rets)/std(trade_rets) * sqrt(252/avg_hold) — check ddof and avg_hold guard.
- min 10 trades filter applied at the (strategy,asset) cell BEFORE aggregation.
- composite weights 0.25 median_return + 0.20 sharpe + 0.20 consistency + 0.15 avg_pf + 0.10 avg_expectancy + 0.10 positive_count/9, each min-max normalized across strategies.
- alpha_vs_bh = total_return - buy&hold over the SAME window.
Hunt for: total_return compounding vs summation (spec wants non-compounded sum), profit_factor inf handling skewing the mean, normalization including inf/NaN, win_rate counting zero-pnl trades, B&H window mismatch.`,
  },
  {
    key: 'validation',
    files: `${SRC}\\\\oos_validator.py, ${SRC}\\\\random_benchmark.py, ${SRC}\\\\sensitivity.py`,
    focus: `Overfitting controls. Verify against spec section 7:
- IS 2022-01..2024-06, OOS 2024-07..2026-06; survival: OOS return>0, valid: avg PF>1, robust: OOS sharpe>0.5*IS sharpe.
- walk-forward 5 windows exactly as the spec table; >=3/5 positive = validated.
- random benchmark: SAME per-asset trade frequency, random entry dates, 1000 sims, real return > 95th pctile => significant; p_value = fraction of sims >= actual.
- sensitivity sweeps period/SL/TP/hold; stable if within +/-30%.
Hunt for: IS/OOS window leakage or overlap, random benchmark using strategy signals instead of random dates, p_value direction inverted, sensitivity not actually varying the signal period (signal_params threading), random sims reusing the same RNG draws.`,
  },
  {
    key: 'strategy_exits_filters',
    files: `${SRC}\\\\strategy.py, ${SRC}\\\\exits.py, ${SRC}\\\\filters.py`,
    focus: `Strategy generator + exit/filter libraries. Verify against spec section 4 and the E1-E12 / F1-F10 tables:
- Layer 1 every (signal,direction) with base exit; Layer 2 the 5 variants; Layer 3 filters on IS-top; Layer 4 pairwise AND same-direction; Layer 5 SEQ short+long.
- exit variants: base(-5/+10/10d), cons(-3/+5/7d), aggr(-7/+15/14d), trail(-3% trail/14d), quick(first-profit+-3%/5d), atr(2xATR SL/3xATR TP/10d).
- AND combo entries = intersection; filters AND-ed; contradictory trend filters (LONG+F2, SHORT+F1) skipped.
- F1-F10 conditions match the table; F8/F9 are cross-asset.
Hunt for: build_entries applying filters to the wrong side, opposite-signal series wiring for E2, combo direction handling, exit compose() dropping a field, filter sign errors (F2 downtrend, F6 low-vol).`,
  },
]

const FINDINGS = {
  type: 'object',
  required: ['dimension', 'findings'],
  properties: {
    dimension: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['severity', 'file', 'description', 'suggested_fix'],
        properties: {
          severity: { type: 'string', enum: ['critical', 'major', 'minor'] },
          file: { type: 'string' },
          location: { type: 'string', description: 'function / line hint' },
          description: { type: 'string' },
          suggested_fix: { type: 'string' },
        },
      },
    },
    notes: { type: 'string' },
  },
}

const VERDICT = {
  type: 'object',
  required: ['is_real', 'confidence', 'reasoning'],
  properties: {
    is_real: { type: 'boolean' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    severity: { type: 'string', enum: ['critical', 'major', 'minor'] },
    reasoning: { type: 'string' },
    corrected_fix: { type: 'string', description: 'the fix to apply if real (may refine the original)' },
  },
}

const reviewPrompt = (d) => `You are an adversarial code reviewer. Find correctness and look-ahead defects in this part of a crypto backtesting engine. Assume bugs exist.

FILES: ${d.files}
SPEC: ${SPEC} (read the relevant sections)

FOCUS:
${d.focus}

Read every file listed and the relevant spec sections. Report EVERY real defect with severity, exact location, a clear description of why it is wrong, and a concrete suggested fix. Do not report style nits. If you find nothing wrong in an area, say so. Return the structured findings.`

const verifyPrompt = (f) => `You are a skeptical verifier. A reviewer claims this is a bug in a crypto backtester. Determine if it is REAL by reading the actual code and spec. Default to is_real=false unless you can demonstrate the defect concretely.

CLAIM (${f.severity}) in ${f.file} ${f.location || ''}:
${f.description}
Proposed fix: ${f.suggested_fix}

Read ${f.file} and ${SPEC}. Trace the actual code path. Confirm or refute. If real, give the corrected fix (refine the proposal if needed).`

phase('Review')
const reviews = await pipeline(
  DIMS,
  (d) => agent(reviewPrompt(d), { label: `review:${d.key}`, phase: 'Review', schema: FINDINGS }),
  (rev, d) => {
    const fs = (rev && rev.findings) || []
    if (!fs.length) return { dimension: d.key, confirmed: [] }
    return parallel(fs.map(f => () =>
      agent(verifyPrompt(f), { label: `verify:${d.key}:${(f.severity || '?')[0]}`, phase: 'Verify', schema: VERDICT })
        .then(v => ({ ...f, verdict: v }))
    )).then(vs => ({ dimension: d.key, confirmed: vs.filter(Boolean).filter(x => x.verdict && x.verdict.is_real) }))
  },
)

const all = reviews.filter(Boolean).flatMap(r => (r.confirmed || []).map(c => ({ dimension: r.dimension, ...c })))
log(`confirmed defects: ${all.length}`)
return {
  confirmed_count: all.length,
  by_severity: {
    critical: all.filter(c => (c.verdict.severity || c.severity) === 'critical').length,
    major: all.filter(c => (c.verdict.severity || c.severity) === 'major').length,
    minor: all.filter(c => (c.verdict.severity || c.severity) === 'minor').length,
  },
  defects: all.map(c => ({
    dimension: c.dimension, file: c.file, location: c.location,
    severity: c.verdict.severity || c.severity,
    description: c.description, fix: c.verdict.corrected_fix || c.suggested_fix,
    confidence: c.verdict.confidence,
  })),
}
