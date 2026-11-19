"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Reads one case per run from the input directory (each <id>.in is a JSON
object, schema "makefun/gta6-trophies-case/v1") and writes one forecast
file per case to the output directory (<id>.out):

    {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}

Five quantiles of each trophy's earn rate (percent of the players
Exophase tracks for the game who have earned it) at the case's target,
scored by mean pinball loss on the logit scale.

How it forecasts
---------------
The page Exophase shows is a snapshot: it only moves when Exophase
refreshes the game's counts, and a refresh both adds a wave of newly
tracked players (dilution: most newcomers have earned almost nothing
yet) and reveals the progress the tracked players made since the last
refresh. So the rate of a trophy is the ratio of two moving totals,
earners over players-tracked, and the forecast models those two totals
separately:

* The history is collapsed to its distinct states (consecutive reads
  that print the same page are one refresh, not new information).
* For the players total and for each trophy's earners, a straight line
  is fitted to the log total against log(1 + hours since release).
  Log time bends growth down as the launch ages, which is what a
  saturating player base and saturating progression look like; the fit
  uses the last 24 hours of states when there are at least three,
  otherwise the whole history. Each trophy's earners slope is shrunk
  toward the cross-trophy median slope, because a single trophy's
  counts are small and its slope is noisy.
* The median is anchored on the trophy's rate in the last read and
  moved by the fitted rate slope (earners slope minus players slope),
  damped by 0.6 and capped, over the log-time gap to the target. The
  target is pulled back by half the median refresh interval, because
  the answer is the page as of the last refresh before the target,
  not the target itself.
* The spread (logit sd) combines the fit's residual, a random-walk
  term growing with the square root of the horizon, and a fixed term
  for refresh-timing uncertainty. A trophy the page prints at 100%
  is forecast near 100% with a narrow spread.

When the history is a single flat page (no refresh seen yet, as on
launch night), none of that can be fitted, so the forecast is a
two-branch mixture: with some probability the page stays as it is,
and otherwise a refresh lands before the target and dilutes every
rate. The dilution factor 0.2207 is the one the house's own rules
document for the launch-night refresh (players 638 -> 3868,
non-100% trophies earned 305 -> 408 in total): a refreshed rate is
the old rate times (408/305) * (638/3868). The refresh probability
compounds with the horizon: a refresh inside a 90-minute window is
put at 0.6 (refreshes come hours apart and the page is already stale
for hours on launch morning), and the same per-window hazard over a
long horizon makes a refresh near-certain. The mixture's quantiles
are the honest quantiles of that two-branch distribution.

Everything is deterministic (no randomness, no wall-clock reads, no
network) and standard library only, so two runs write identical
bytes. Any unexpected problem with a case falls back to a plain
carry-forward forecast rather than crashing, because a missing or
invalid output scores far worse than a mediocre one.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Refresh probability of a 90-minute window when the page has never
# been seen to refresh. Refreshes come hours apart and the page is
# already stale for hours on launch morning, so a refresh inside the
# window is more likely than not; the same per-window hazard
# compounds over longer horizons.
P_REFRESH_90MIN = 0.6
# Documented first-refresh dilution for non-100% trophies (see the
# module docstring): (408 / 305) * (638 / 3868).
DILUTION = 0.2207
# Logit sd of the refresh branch of the flat-history mixture.
REFRESH_SD = 0.4
# Damping on the fitted rate slope when extrapolating: growth
# saturates, so the fitted log-time slope overstates the future move.
SLOPE_DAMP = 0.6
# Cap on the fitted rate slope, in logit units per log-time unit.
SLOPE_CAP = 1.5
# Shrinkage strength for a trophy's earners slope: weight
# n_states / (n_states + SHRINK_K) on its own slope, the rest on the
# cross-trophy median slope.
SHRINK_K = 8.0
# Spread: random-walk coefficient (logit sd per sqrt day) and the
# fixed refresh-timing term, both in logit units.
SPREAD_RW = 0.30
SPREAD_REFRESH = 0.25
SPREAD_FLOOR = 0.15
SPREAD_MIN = 0.35
SPREAD_CAP = 2.0
# Logit sd for a trophy printed at 100%.
FULL_SD = 0.5


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(x):
    return 100.0 / (1.0 + math.exp(-x))


def round4(x):
    """Round to 4 decimals (half up), returning an int for whole numbers."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text, default):
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return default
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def linfit(xs, ys):
    """Least-squares line y ~ a + b*x; returns (a, b, residual sd)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        sd = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    else:
        sd = 0.0
    return a, b, sd


def mix_quantiles(branches, taus):
    """Quantiles of a mixture of logit-normals, by grid CDF inversion.

    branches: list of (weight, mean_logit, sd_logit). Deterministic:
    the grid and the inversion are fixed, so the same branches always
    give the same quantiles.
    """
    lo, hi = -14.0, 7.0
    count = 2001
    grid = [lo + (hi - lo) * i / (count - 1) for i in range(count)]
    root2 = math.sqrt(2.0)
    cdf = []
    for x in grid:
        total = 0.0
        for weight, mu, sd in branches:
            total += weight * 0.5 * (1.0 + math.erf((x - mu) / (sd * root2)))
        cdf.append(total)
    out = []
    for tau in taus:
        j = next((i for i, c in enumerate(cdf) if c >= tau), count - 1)
        if j == 0:
            out.append(grid[0])
        else:
            c0, c1 = cdf[j - 1], cdf[j]
            x0, x1 = grid[j - 1], grid[j]
            out.append(x0 + (x1 - x0) * (tau - c0) / max(1e-12, c1 - c0))
    return [inv_logit(x) for x in out]


def quantiles_from(median_rate, sd):
    centre = logit(median_rate)
    return [round4(inv_logit(centre + z * sd)) for z in Z]


def last_rate(case, index):
    """A trophy's rate in the last read that lists it, else 1.0."""
    for read in reversed(case.get("history", [])):
        rates = read.get("rates") or []
        if index < len(rates) and rates[index] is not None:
            return float(rates[index])
    return 1.0


def carry_forward(case):
    """Fallback forecast: every rate stays where the last read put it,
    with the house baseline's random-walk spread."""
    horizon_days = max(float(case.get("horizon_minutes", 0)) / 1440.0, 0.0)
    spread = 0.5 * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(case.get("trophies", [])):
        rate = last_rate(case, i)
        if rate >= 99.99:
            rows.append({"id": trophy["id"],
                         "q": quantiles_from(99.99, FULL_SD)})
        else:
            rows.append({"id": trophy["id"],
                         "q": quantiles_from(rate, max(spread, 0.05))})
    return {"trophies": rows}


def flat_history_forecast(case):
    """No refresh has been seen: mix 'no refresh' with 'a refresh
    dilutes every rate' (see the module docstring)."""
    horizon_hours = max(float(case.get("horizon_minutes", 0)) / 60.0, 0.0)
    horizon_days = horizon_hours / 24.0
    calm_sd = max(0.5 * math.sqrt(horizon_days), 0.05)
    p_refresh = 1.0 - (1.0 - P_REFRESH_90MIN) ** (horizon_hours / 1.5)
    rows = []
    for i, trophy in enumerate(case.get("trophies", [])):
        rate = last_rate(case, i)
        if rate >= 99.99:
            rows.append({"id": trophy["id"],
                         "q": quantiles_from(99.99, FULL_SD)})
            continue
        branches = [
            (1.0 - p_refresh, logit(rate), calm_sd),
            (p_refresh, logit(max(0.01, rate * DILUTION)), REFRESH_SD),
        ]
        rows.append({"id": trophy["id"],
                     "q": [round4(q) for q in mix_quantiles(branches, LEVELS)]})
    return {"trophies": rows}


def trend_forecast(case, states, release):
    """History has refreshes: fit log earners and log players against
    log time, anchor on the last read, extrapolate to the target."""
    horizon_hours = max(float(case.get("horizon_minutes", 0)) / 60.0, 0.0)

    def x_of(moment):
        hours = max((moment - release).total_seconds() / 3600.0, 0.0)
        return math.log1p(hours)

    xs = []
    log_players = []
    hours_since_release = []
    for read in states:
        moment = parse_time(read.get("at"), release)
        xs.append(x_of(moment))
        players = read.get("players_tracked") or 1
        log_players.append(math.log(max(float(players), 1.0)))
        hours_since_release.append((moment - release).total_seconds() / 3600.0)

    gaps = [hours_since_release[k] - hours_since_release[k - 1]
            for k in range(1, len(hours_since_release))]
    interval = median(gaps) if gaps else 8.0
    interval = min(max(interval, 1.0), 48.0)

    target = parse_time(case.get("target"), release)
    x_target = x_of(target - timedelta(hours=interval / 2.0))

    # Fit on the last 24 hours of states when there are at least
    # three of them; otherwise on the whole history.
    recent = [k for k in range(len(xs))
              if hours_since_release[k] >= hours_since_release[-1] - 24.0]
    window = recent if len(recent) >= 3 else list(range(len(xs)))

    a_p, b_p, _ = linfit([xs[k] for k in window],
                         [log_players[k] for k in window])

    fits = []
    rate_slopes = []
    for i, trophy in enumerate(case.get("trophies", [])):
        xe = []
        ye = []
        for k in window:
            read = states[k]
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                earners = max(float(rates[i]) * float(read.get("players_tracked") or 1) / 100.0, 0.5)
                xe.append(xs[k])
                ye.append(math.log(earners))
        if len(xe) >= 2:
            a_e, b_e, sd_e = linfit(xe, ye)
        else:
            a_e = ye[0] if ye else 0.0
            b_e = b_p
            sd_e = 0.0
        fits.append((a_e, b_e, sd_e))
        rate_slopes.append(b_e - b_p)
    median_rate_slope = median(rate_slopes)

    x_last = xs[-1]
    dx = x_target - x_last
    rows = []
    for i, trophy in enumerate(case.get("trophies", [])):
        rate = last_rate(case, i)
        if rate >= 99.99:
            rows.append({"id": trophy["id"],
                         "q": quantiles_from(99.99, FULL_SD)})
            continue
        _, b_e, sd_e = fits[i]
        weight = len(states) / (len(states) + SHRINK_K)
        b_shrunk = weight * b_e + (1.0 - weight) * (b_p + median_rate_slope)
        slope = max(-SLOPE_CAP, min(SLOPE_CAP, b_shrunk - b_p))
        centre = logit(rate) + slope * SLOPE_DAMP * dx
        median_rate = min(max(100.0 / (1.0 + math.exp(-centre)), 0.01), 99.99)
        spread = math.sqrt(max(sd_e, SPREAD_FLOOR) ** 2
                           + (SPREAD_RW * math.sqrt(horizon_hours / 24.0)) ** 2
                           + SPREAD_REFRESH ** 2)
        spread = min(max(spread, SPREAD_MIN), SPREAD_CAP)
        rows.append({"id": trophy["id"],
                     "q": quantiles_from(median_rate, spread)})
    return {"trophies": rows}


def forecast_case(case):
    trophies = case.get("trophies", [])
    history = case.get("history", [])
    if not trophies:
        return {"trophies": []}
    if not history:
        return carry_forward(case)
    # Collapse the history to its distinct states: consecutive reads
    # printing the same page are one refresh, not new information.
    states = []
    for read in history:
        key = (read.get("players_tracked"),
               tuple(read.get("rates") or []))
        if not states or states[-1][1] != key:
            states.append((read, key))
    states = [read for read, _ in states]
    if len(states) == 1:
        return flat_history_forecast(case)
    release_text = (case.get("game") or {}).get("release_date", "2026-11-19")
    release = parse_time(release_text, datetime(2026, 11, 19, tzinfo=timezone.utc))
    return trend_forecast(case, states, release)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                case = {"trophies": [], "history": [], "horizon_minutes": 0}
            forecast = carry_forward(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
