#!/usr/bin/env python3
"""Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, 240 minutes ahead.

Method (per market, from that market's own price history only):

  * Median: asymmetric mean reversion. The median drifts from the last price
    back toward the recent level (the mean of the last LEVEL_WINDOW minutes).
    Up-spikes revert hard; down-moves mostly stick. Reversion strength grows
    with horizon and the adjustment is shrunk by SHRINK.
  * Bands: empirical quantiles of the market's own past h-step price changes,
    recency-weighted (decay RECENCY_TAU minutes), computed on a coarse
    horizon grid and interpolated in log-horizon, then centred on the
    reversion median. No Gaussian assumption: the tails are whatever the
    market actually did.
  * Floor: the 90% band half-width never falls below a floor that grows
    slowly with horizon, so quiet markets still carry honest uncertainty.
  * Fallbacks: pooled cross-market bands (scaled by the market's own
    1-minute volatility) when a market has too little history; the house
    persistence baseline if anything at all fails for a market.

Stdlib only. Deterministic: no randomness, fixed iteration order, so two
runs write the same bytes.
"""

import bisect
import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.644854, -0.67449, 0.0, 0.67449, 1.644854)
BASE_SIGMA = 0.0025

# --- tunable parameters, calibrated on the six practice cases (see notes) ---
# Calibration: coordinate descent on practice pinball loss, 2026-11-13.
# Beats the house baseline (0.363589) on every practice case; overall 0.283.
LEVEL_WINDOW = 90           # minutes averaged for the reversion level
SHRINK = 0.6                # weight on the reversion adjustment
RECENCY_TAU = 100.0         # minutes; exponential decay of window weights
GRID = (1, 2, 3, 5, 8, 12, 18, 27, 40, 60, 90, 130, 180, 240)
# Reversion strength by grid horizon, up-spikes and down-moves separately.
# Up-spikes revert hard and increasingly with horizon; down-moves stick
# fully (zero reversion won the calibration outright: a price that fell
# stays fallen far more often than it bounces).
BETA_UP = (0.1725, 0.2875, 0.345, 0.46, 0.552, 0.621, 0.69, 0.782,
           0.8625, 0.92, 0.943, 0.9545, 0.966, 0.9775)
BETA_DOWN = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
FLOOR_240 = 0.002           # minimum 90% half-width at h=240
FLOOR_EXP = 0.37            # floor grows like h**FLOOR_EXP
MIN_WINDOWS = 50            # windows needed to trust a market's own bands
HISTORY_CAP = 1200          # only windows ending within the last N minutes


def clamp6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1000000) / 1000000 + 0.0


def _interp_log_h(grid, vals, h):
    """Interpolate vals (given at grid horizons) at horizon h, in log-h."""
    if h <= grid[0]:
        return vals[0]
    if h >= grid[-1]:
        return vals[-1]
    lh = math.log(h)
    for i in range(len(grid) - 1):
        if grid[i] <= h <= grid[i + 1]:
            lo = math.log(grid[i])
            hi = math.log(grid[i + 1])
            f = (lh - lo) / (hi - lo) if hi > lo else 0.0
            return vals[i] + (vals[i + 1] - vals[i]) * f
    return vals[-1]


def _weighted_quantiles(pairs, taus):
    """Weighted quantiles of (value, weight) pairs, linearly interpolated."""
    pairs = sorted(pairs)
    cum = []
    total = 0.0
    for _, w in pairs:
        total += w
        cum.append(total)
    out = []
    for tau in taus:
        target = tau * total
        i = bisect.bisect_left(cum, target)
        if i <= 0:
            out.append(pairs[0][0])
        elif i >= len(pairs):
            out.append(pairs[-1][0])
        else:
            c0, c1 = cum[i - 1], cum[i]
            v0, v1 = pairs[i - 1][0], pairs[i][0]
            f = (target - c0) / (c1 - c0) if c1 > c0 else 0.0
            out.append(v0 + (v1 - v0) * f)
    return out


def _median_path(prices, horizon):
    """Reversion median for h = 1..horizon."""
    n = len(prices)
    last = prices[-1]
    w = min(LEVEL_WINDOW, n)
    level = sum(prices[n - w:]) / w
    dev = last - level
    betas = BETA_UP if dev >= 0 else BETA_DOWN
    path = []
    for h in range(1, horizon + 1):
        b = _interp_log_h(GRID, betas, h)
        path.append(min(1.0, max(0.0, last - SHRINK * b * dev)))
    return path


def _one_min_sd(prices):
    if len(prices) < 3:
        return 0.0
    diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    m = sum(diffs) / len(diffs)
    var = sum((d - m) ** 2 for d in diffs) / len(diffs)
    return math.sqrt(var)


def _change_pairs(prices, g):
    """Recency-weighted (change, weight) pairs for g-step moves."""
    n = len(prices)
    first = max(0, n - 1 - g - HISTORY_CAP)
    pairs = []
    for t in range(first, n - g):
        age = (n - 1) - (t + g)
        pairs.append((prices[t + g] - prices[t],
                      math.exp(-age / RECENCY_TAU)))
    return pairs


def _build_pooled(all_prices):
    """Pooled centred spreads per grid horizon + pooled 1-minute sd."""
    pooled = {}
    for g in GRID:
        pairs = []
        for prices in all_prices:
            pairs.extend(_change_pairs(prices, g))
        if len(pairs) >= MIN_WINDOWS:
            qs = _weighted_quantiles(pairs, TAUS)
            pooled[g] = [q - qs[2] for q in qs]
        else:
            pooled[g] = None
    diffs = []
    for prices in all_prices:
        diffs.extend(prices[i + 1] - prices[i]
                     for i in range(len(prices) - 1))
    pooled_sd = None
    if len(diffs) >= 2:
        m = sum(diffs) / len(diffs)
        pooled_sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / len(diffs))
    return pooled, pooled_sd


def _vol_scale(prices, pooled_sd):
    """Scale pooled bands to this market's own 1-minute volatility."""
    if pooled_sd is None or pooled_sd <= 0:
        return 1.0
    own = _one_min_sd(prices)
    if own <= 0:
        return 0.25
    return min(3.0, max(0.1, own / pooled_sd))


def _market_spreads(prices, pooled, pooled_sd):
    """Centred empirical spreads per grid horizon for one market."""
    spreads = {}
    for g in GRID:
        pairs = _change_pairs(prices, g)
        if len(pairs) >= MIN_WINDOWS:
            qs = _weighted_quantiles(pairs, TAUS)
            spreads[g] = [q - qs[2] for q in qs]
        else:
            spreads[g] = None
    if any(v is None for v in spreads.values()) and pooled is not None:
        scale = _vol_scale(prices, pooled_sd)
        for g in GRID:
            if spreads[g] is None and pooled[g] is not None:
                spreads[g] = [s * scale for s in pooled[g]]
    for g in GRID:
        if spreads[g] is None:
            s = BASE_SIGMA * math.sqrt(g)
            spreads[g] = [z * s for z in Z]
    return spreads


def _floor(h):
    return FLOOR_240 * (h / 240.0) ** FLOOR_EXP


def _forecast_rows(prices, spreads, horizon):
    med = _median_path(prices, horizon)
    grid_spreads = [[spreads[g][j] for g in GRID] for j in range(5)]
    rows = []
    for h in range(1, horizon + 1):
        m = med[h - 1]
        q = [m + _interp_log_h(GRID, grid_spreads[j], h) for j in range(5)]
        hw = (q[4] - q[0]) / 2.0
        fl = _floor(h)
        if hw < fl:
            if hw > 1e-12:
                f = fl / hw
                q = [m + (v - m) * f for v in q]
            else:
                q = [m + z * fl / 1.644854 for z in Z]
        q = [clamp6(v) for v in q]
        for j in range(1, 5):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        rows.append(q)
    return rows


def _baseline_rows(last, horizon):
    """House persistence baseline: last price held flat, sqrt(h) bands."""
    rows = []
    for h in range(1, horizon + 1):
        s = BASE_SIGMA * math.sqrt(h)
        rows.append([clamp6(last + z * s) for z in Z])
    return rows


def _forecast_market(market, pooled, pooled_sd, horizon):
    prices = market.get("prices") or []
    if not prices:
        return _baseline_rows(0.5, horizon)
    spreads = _market_spreads(prices, pooled, pooled_sd)
    return _forecast_rows(prices, spreads, horizon)


def run_case(case):
    horizon = case["horizon_minutes"]
    markets = case["markets"]
    need_pooled = any(
        len(m.get("prices") or []) < MIN_WINDOWS + GRID[-1] + 1
        for m in markets)
    pooled = pooled_sd = None
    if need_pooled:
        pooled, pooled_sd = _build_pooled(
            [m.get("prices") or [] for m in markets])
    forecasts = []
    for m in markets:
        try:
            q = _forecast_market(m, pooled, pooled_sd, horizon)
        except Exception:
            prices = m.get("prices") or []
            last = prices[-1] if prices else 0.5
            q = _baseline_rows(last, horizon)
        forecasts.append({"id": m["id"], "q": q})
    return {"forecasts": forecasts}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in files:
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        out = run_case(case)
        with open(os.path.join(OUTPUT_DIR, name[:-3] + ".out"),
                  "w", encoding="utf-8") as fh:
            fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
