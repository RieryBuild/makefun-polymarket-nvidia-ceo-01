#!/usr/bin/env python3
"""Entry for make.fun's polymarket-nvidia-ceo-forecast-01.

Forecasts, minute by minute and 4 hours ahead, the Yes price of every trading
market of Polymarket's "Next CEO of NVIDIA?" event, as five quantiles
[0.05, 0.25, 0.5, 0.75, 0.95] per minute, minimising the mean pinball loss.

Design (all three parts were found in the practice history, then verified by
backtesting on all six practice cases against answers reconstructed from the
cases themselves and from Polymarket's public price history):

1. Median: persistence + ASYMMETRIC mean reversion. A positive 60-minute trend
   reverts hard (regression beta 0.21 at h=5 growing to 0.79 at h=240); a
   negative trend sticks (beta ~ 0). Up-spikes in these thin markets are
   mostly quote noise that fades; down-moves are real repricing. The fitted
   beta is shrunk to 85% for robustness.
2. Spread: EMPIRICAL quantiles of h-minute price changes, per market, computed
   at a grid of horizons and interpolated in log-h, recentered on the modeled
   median. The change distribution has a big atom at zero (71-97% of minutes
   are flat) and fat, asymmetric tails, so normal bands are far too wide in
   the middle and mis-shaped in the tails. Changes are recency-weighted
   (exponential decay, tau = 150 min) so the bands track the current regime.
3. Floor: a minimum half-width growing like h^0.37, so a market flat for a day
   does not get zero-width bands the minute it wakes up.

Backtest on the six practice cases (answers reconstructed exactly: the
baseline starter reproduces its published 0.363589 on the same scorer):
  this entry  loss 0.270340, median error 0.729894, 90% coverage 0.8855
  house baseline loss 0.363589, median error 0.767503, 90% coverage 0.9496
It beats the baseline on every practice case (p00 0.456, p01 0.305, p02 0.350,
p03 0.291, p04 0.170, p05 0.114) and stays ahead of it under synthetic jump
stress tests (sustained +0.15 / -0.10 moves injected into the answers).

Standard library only, no network, deterministic (no randomness, no clocks).
Runs a 24-market case in ~1 second (cap: 8 s).
"""
import json, math, os

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
H_GRID = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180, 240]

DEFAULTS = dict(
    window=4320,        # minutes of history used for empirical spreads (72 h)
    trend_look=60,      # minutes for the reversion trend
    beta_up_a=-0.027,   # beta_up(h) = clip(a + b*ln(h)) * shrink
    beta_up_b=0.149,
    beta_up_shrink=0.85,
    beta_up_max=0.80,
    beta_dn=0.0,        # down-trend coefficient (0 = persistence; down-moves stick)
    floor_hw240=0.004,  # minimum half-width (q95-q50) at h=240
    floor_exp=0.37,     # horizon exponent of the floor
    pool_blend=0.0,     # weight of pooled spreads (0 = per-market only)
    min_hist=30,        # below this, fall back to pooled spreads
    decay=150,          # recency-decay tau (minutes) for empirical spreads
    trend_mode="point",  # "point": x - p(t-look); "median": x - median(last look prices)
    tail_mult=1.0,      # multiplier on the 5%/95% half-widths only (tail insurance)
)


def _quantile(sorted_vals, q):
    """Linear-interpolation quantile of a sorted list."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


_EXPW = {}  # decay tau -> precomputed exp(-age/tau) weights, age 0..65535


def _exp_weights(tau, n):
    w = _EXPW.get(tau)
    if w is None or len(w) < n:
        w = [math.exp(-a / tau) for a in range(max(n, 4321))]
        _EXPW[tau] = w
    return w


def _weighted_quantiles(pairs, taus):
    """All quantiles of values with weights in ONE sorted pass.

    pairs: list of (value, weight). Returns [q for q in taus].
    """
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        mid = pairs[len(pairs) // 2][0]
        return [mid] * len(taus)
    targets = [t * total for t in taus]
    out = [None] * len(taus)
    acc = 0.0
    ti = 0
    prev_v = pairs[0][0]
    for v, w in pairs:
        acc += w
        while ti < len(taus) and acc >= targets[ti]:
            overshoot = (acc - targets[ti]) / w if w > 0 else 0.0
            out[ti] = v * (1 - overshoot) + prev_v * overshoot
            ti += 1
        prev_v = v
    for j in range(len(taus)):
        if out[j] is None:
            out[j] = pairs[-1][0]
    return out


def _empirical_spreads(prices, window, h_grid, decay=0):
    """Per-horizon empirical quantiles of h-minute changes, recentered.

    decay: if > 0, recency-weight each change window by exp(-age/decay),
    age in minutes from the most recent window. The window is truncated to
    6*decay (weights beyond that are < 0.25% and cost compute).
    Returns {h: [s05..s95]} with s_j = q_j - q_50.
    """
    if decay and decay > 0:
        window = min(window, 6 * decay)
    p = prices[-window:] if len(prices) > window else prices
    n = len(p)
    out = {}
    for h in h_grid:
        if n <= h:
            continue
        if decay and decay > 0:
            w = _exp_weights(decay, n - h)
            pairs = [(p[i + h] - p[i], w[(n - h - 1) - i]) for i in range(n - h)]
            qs = _weighted_quantiles(pairs, TAUS)
        else:
            ch = sorted(p[i + h] - p[i] for i in range(n - h))
            qs = [_quantile(ch, t) for t in TAUS]
        med = qs[2]
        out[h] = [q - med for q in qs]
    return out


def _interp_spreads(spreads, h):
    """Interpolate recentered spreads at horizon h (linear in log h)."""
    hs = sorted(spreads)
    if not hs:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    if h <= hs[0]:
        return spreads[hs[0]]
    if h >= hs[-1]:
        return spreads[hs[-1]]
    for i in range(len(hs) - 1):
        if hs[i] <= h <= hs[i + 1]:
            lo, hi = hs[i], hs[i + 1]
            w = (math.log(h) - math.log(lo)) / (math.log(hi) - math.log(lo))
            a, b = spreads[lo], spreads[hi]
            return [a[j] * (1 - w) + b[j] * w for j in range(5)]
    return spreads[hs[-1]]


def _beta_up(h, P):
    b = (P["beta_up_a"] + P["beta_up_b"] * math.log(max(h, 1))) * P["beta_up_shrink"]
    return max(0.0, min(P["beta_up_max"], b))


def _baseline_rows(x, horizon):
    """House-baseline persistence forecast around x (fallback; never crashes)."""
    z = [-1.644854, -0.67449, 0, 0.67449, 1.644854]
    rows = []
    for h in range(1, horizon + 1):
        s = 0.0025 * math.sqrt(h)
        q = [round(min(1.0, max(0.0, x + zi * s)), 6) + 0.0 for zi in z]
        for j in range(1, 5):
            q[j] = max(q[j], q[j - 1])
        rows.append(q)
    return rows


def forecast_market(prices, horizon, P, pooled_spreads=None):
    """Forecast rows for one market. prices: list of floats, last = checkpoint."""
    if not prices:
        return _baseline_rows(0.5, horizon)
    x = prices[-1]
    n = len(prices)
    # --- median path: asymmetric reversion of the recent trend ---
    look = P["trend_look"]
    if P.get("trend_mode") == "median" and n > 1:
        seg = sorted(prices[-look:] if n >= look else prices)
        base = seg[len(seg) // 2]
        trend = x - base
    elif n > look:
        trend = x - prices[-1 - look]
    elif n > 1:
        trend = x - prices[0]
    else:
        trend = 0.0
    # --- spreads ---
    if n >= P["min_hist"]:
        spreads = _empirical_spreads(prices, P["window"], H_GRID, P.get("decay", 0))
    else:
        spreads = {}
    use_pooled = pooled_spreads is not None and (not spreads or P["pool_blend"] > 0)
    rows = []
    for h in range(1, horizon + 1):
        if trend > 0:
            med = x - _beta_up(h, P) * trend
        else:
            med = x - P["beta_dn"] * trend
        med = min(1.0, max(0.0, med))
        if spreads:
            sp = _interp_spreads(spreads, h)
            if use_pooled and P["pool_blend"] > 0:
                pp = _interp_spreads(pooled_spreads, h)
                w = P["pool_blend"]
                sp = [sp[j] * (1 - w) + pp[j] * w for j in range(5)]
        elif pooled_spreads:
            sp = _interp_spreads(pooled_spreads, h)
        else:
            sp = [-0.0025 * 1.644854 * math.sqrt(h), -0.0025 * 0.67449 * math.sqrt(h),
                  0.0, 0.0025 * 0.67449 * math.sqrt(h), 0.0025 * 1.644854 * math.sqrt(h)]
        # floor: minimum half-width, growing with horizon
        fl = P["floor_hw240"] * (h / 240.0) ** P["floor_exp"]
        tm = P.get("tail_mult", 1.0)
        lo_w = max(-sp[0], fl) * tm   # downside half-width at 5%
        hi_w = max(sp[4], fl) * tm    # upside half-width at 95%
        # scale the inner quantiles to at least the floor shape too
        q = [med - lo_w,
             med - max(-sp[1], fl * 0.41),
             med,
             med + max(sp[3], fl * 0.41),
             med + hi_w]
        # clamp, round, enforce non-decreasing
        q = [min(1.0, max(0.0, v)) for v in q]
        q = [round(v, 6) + 0.0 for v in q]
        for j in range(1, 5):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        rows.append(q)
    return rows


def pooled_spreads_all(markets, P):
    """Pooled empirical spreads across markets (for thin-market fallback)."""
    pooled = {}
    for h in H_GRID:
        ch = []
        for m in markets:
            p = m["prices"][-P["window"]:] if len(m["prices"]) > P["window"] else m["prices"]
            if len(p) > h:
                ch.extend(p[i + h] - p[i] for i in range(len(p) - h))
        if not ch:
            continue
        ch.sort()
        qs = [_quantile(ch, t) for t in TAUS]
        med = qs[2]
        pooled[h] = [q - med for q in qs]
    return pooled


def forecast_case(case, P=None):
    P = dict(DEFAULTS, **(P or {}))
    horizon = case["horizon_minutes"]
    markets = case["markets"]
    pooled = pooled_spreads_all(markets, P) if any(len(m["prices"]) < P["min_hist"] for m in markets) or P["pool_blend"] > 0 else None
    out = []
    for m in markets:
        try:
            rows = forecast_market(m["prices"], horizon, P, pooled)
        except Exception:
            # One weird market must never kill the case: fall back to the
            # house baseline for it (an invalid case scores as all-0.5).
            x = m["prices"][-1] if m["prices"] else 0.5
            rows = _baseline_rows(x, horizon)
        out.append({"id": m["id"], "q": rows})
    return {"forecasts": out}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for f in sorted(os.listdir(in_dir)):
        if not f.endswith(".in"):
            continue
        cid = f[:-3]
        with open(os.path.join(in_dir, f)) as fh:
            case = json.load(fh)
        fc = forecast_case(case)
        with open(os.path.join(out_dir, cid + ".out"), "w") as fh:
            json.dump(fc, fh)
            fh.write("\n")


if __name__ == "__main__":
    main()
