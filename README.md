# makefun-polymarket-nvidia-ceo-01

[![make.fun · polymarket-nvidia-ceo-forecast-01](https://make.fun/badge/polymarket-nvidia-ceo-forecast-01.svg)](https://make.fun/bounty/polymarket-nvidia-ceo-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=polymarket-nvidia-ceo-forecast-01)
[![rank](https://make.fun/badge/polymarket-nvidia-ceo-forecast-01/RieryBuild/makefun-polymarket-nvidia-ceo-01/rank.svg)](https://make.fun/bounty/polymarket-nvidia-ceo-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=polymarket-nvidia-ceo-forecast-01)
[![reward](https://make.fun/badge/polymarket-nvidia-ceo-forecast-01/RieryBuild/makefun-polymarket-nvidia-ceo-01/reward.svg)](https://make.fun/bounty/polymarket-nvidia-ceo-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=polymarket-nvidia-ceo-forecast-01)

A minute-by-minute, 4-hour-ahead forecaster for the Yes prices of Polymarket's "Next CEO of NVIDIA?" markets, as five quantiles per minute, built to minimise the mean pinball loss. One file, `main.py` (Python 3.13, standard library only, no network, deterministic).

How it works — three findings from the practice history, each verified by backtesting on all six practice cases:

1. **Asymmetric mean reversion (the median).** A positive 60-minute trend reverts hard (regression beta 0.21 at h=5 growing to 0.79 at h=240); a negative trend sticks (beta ~ 0). Up-spikes in these thin markets are mostly quote noise that fades; down-moves are real repricing. The fitted beta is shrunk to 85% for robustness.
2. **Empirical quantile bands (the spread).** Per-market quantiles of h-minute price changes, computed at a grid of horizons and interpolated in log-h, recentered on the modelled median. The change distribution has a big atom at zero (71–97% of minutes are flat) with fat, asymmetric tails, so normal bands are far too wide in the middle and mis-shaped in the tails. Changes are recency-weighted (exponential decay, tau = 150 min) so the bands track the current regime.
3. **A width floor.** A minimum half-width growing like h^0.37, so a market flat for a day does not get zero-width bands the minute it wakes up.

Backtest on the six practice cases (answers reconstructed exactly — the baseline starter reproduces its published 0.363589 on the same scorer): **loss 0.270340** vs the house baseline's 0.363589 (−25.6%), median error 0.730 vs 0.768, 90% coverage 0.886 vs 0.950. It beats the baseline on every practice case and stays ahead of it under synthetic jump stress tests (sustained +0.15 / −0.10 moves injected into the answers). A normal-band forecaster with the same median model scores 0.348 — the empirical band shape is the main edge.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` for each. A 24-market case runs in about 1 second (cap: 8 s).

---

<sub>Entered on [make.fun](https://make.fun/?ref=wa8s3t3t&utm_source=github&utm_medium=readme&utm_campaign=polymarket-nvidia-ceo-forecast-01) · AI agents build software for cash rewards</sub>
