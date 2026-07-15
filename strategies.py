"""
strategies.py
-------------
Given what we know at time t, decide what to do, then measure how it turned out.

A "position" is our bet on where the ratio goes next:
   +1  = long the ratio  (we profit if the ratio goes UP)
   -1  = short the ratio (we profit if the ratio goes DOWN)
    0  = stay out (no bet)

Profit for one step is simple and intuitive:

    profit = position * (next_ratio - current_ratio) / current_ratio

If we bet up (+1) and the ratio rose, profit is positive. If we bet up and it
fell, we lose. Betting down (-1) flips the sign. Staying out (0) earns nothing.
This is a simplified return and ignores real-world costs like fees and slippage,
which keeps the focus on the strategy logic.

Each strategy is just a rule that returns a position given the recent numbers.
A small threshold means "only trade when the signal is big enough to bother."
"""


def mean_reversion(prev_ratio, current_ratio, predicted_ratio, threshold):
    """
    Bet that recent moves reverse.

    If the ratio just went UP, we expect it to come back DOWN, so we short (-1).
    If it just went DOWN, we expect a bounce, so we go long (+1).
    Ignores the forecast entirely -- it only looks at the last real move.
    """
    recent_move = current_ratio - prev_ratio
    if recent_move > threshold:
        return -1
    if recent_move < -threshold:
        return +1
    return 0


def pure_forecasting(prev_ratio, current_ratio, predicted_ratio, threshold):
    """
    Bet in the direction the forecaster predicts.

    If the model thinks the ratio will rise, go long (+1); if it thinks the
    ratio will fall, go short (-1). Ignores the recent move -- it trusts the
    model.
    """
    predicted_move = predicted_ratio - current_ratio
    if predicted_move > threshold:
        return +1
    if predicted_move < -threshold:
        return -1
    return 0


def hybrid(prev_ratio, current_ratio, predicted_ratio, threshold):
    """
    Only trade when reversion and the forecast AGREE.

    - Ratio just fell AND model predicts a rise  -> go long (+1)
    - Ratio just rose AND model predicts a fall  -> go short (-1)
    - Otherwise stay out. Fewer trades, but each has two reasons behind it.
    """
    recent_move = current_ratio - prev_ratio
    predicted_move = predicted_ratio - current_ratio

    if recent_move < -threshold and predicted_move > threshold:
        return +1
    if recent_move > threshold and predicted_move < -threshold:
        return -1
    return 0


# Lookup table so the command-line tool can pick a strategy by name.
STRATEGIES = {
    "reversion": mean_reversion,
    "forecasting": pure_forecasting,
    "hybrid": hybrid,
}


def run_strategy(strategy_fn, ratios, predictions, threshold):
    """
    Walk through the test period one step at a time, apply the strategy rule,
    and record the profit of each trade.

    Inputs:
      strategy_fn  - one of the functions above
      ratios       - the TRUE ratio values over the test period
      predictions  - the forecaster's guess for each next value (same length)
      threshold    - minimum signal size needed to place a trade

    Returns a dict of results (profits per step, total, number of trades, etc.).
    """
    profits = []

    # Start at index 1 so we always have a "previous" ratio to look back at,
    # and stop one early so there is always a "next" ratio to score against.
    for t in range(1, len(ratios) - 1):
        prev_ratio = ratios[t - 1]
        current_ratio = ratios[t]
        predicted_ratio = predictions[t]
        next_ratio = ratios[t + 1]

        position = strategy_fn(prev_ratio, current_ratio, predicted_ratio, threshold)
        step_profit = position * (next_ratio - current_ratio) / current_ratio
        profits.append(step_profit)

    trades = [p for p in profits if p != 0]
    wins = [p for p in trades if p > 0]

    return {
        "profits": profits,
        "total_profit": sum(profits),
        "num_trades": len(trades),
        "win_rate": (len(wins) / len(trades)) if trades else 0.0,
    }
