# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Mittominato/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Mittominato/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, its earn rate on Exophase at 2026-11-26T00:00:00Z — the percent of the players Exophase tracks who will have earned it a week after launch — as five quantiles per trophy (5%, 25%, 50%, 75%, 95%), scored by mean pinball loss on the logit scale.

## How it forecasts

Exophase only moves its counts when it refreshes a game's page, and a refresh both adds a wave of newly tracked players (dilution: newcomers have earned almost nothing yet) and reveals the progress tracked players made since the last refresh. So a trophy's rate is the ratio of two moving totals — earners over players tracked — and `main.py` models those two totals separately:

- The history is collapsed to its distinct states (consecutive reads printing the same page are one refresh, not new information).
- For the players total and for each trophy's earners, a straight line is fitted to the log total against log(1 + hours since release), using the last 24 hours of states when there are at least three. Each trophy's earners slope is shrunk toward the cross-trophy median slope, because one trophy's counts are small and its slope is noisy.
- The median is anchored on the trophy's rate in the last read and moved by the fitted rate slope (earners slope minus players slope), damped by 0.6 and capped, over the log-time gap to the target. The target is pulled back by half the median refresh interval, because the answer is the page as of the last refresh before the target.
- The spread combines the fit's residual, a random-walk term growing with the square root of the horizon, and a fixed term for refresh-timing uncertainty.

When the history is a single flat page (no refresh seen yet, as on launch night), the forecast is a two-branch mixture: with some probability the page stays as it is, and otherwise a refresh lands before the target and dilutes every rate by the factor the house's own rules document for the launch-night refresh (players 638 → 3868, non-100% trophies earned 305 → 408 in total).

On the kit's practice cases (launch night, both straddling the first refresh) it scores a pinball loss of 0.341, against 1.508 for the house carry-forward baseline. On simulated final cases (two days of refresh history, 4.5-day horizon) it scores 0.16–0.19 across five progression/dilution scenarios, against 0.25–0.28 for the house curve baseline.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is Python 3.13, standard library only, deterministic, and runs offline: it reads every `*.in` case in the input directory and writes one `<id>.out` forecast per case to the output directory.

---

<sub>Entered on [make.fun](https://make.fun/?ref=6xedf17w&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
