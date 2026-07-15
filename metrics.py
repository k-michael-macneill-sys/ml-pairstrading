"""
metrics.py
----------
Small, self-contained functions to score how good a forecast or a strategy is.
Each one takes plain lists of numbers and returns a single number.
"""

import math


def rmse(actual, predicted):
    """
    Root Mean Squared Error: the typical size of the forecast's mistake.
    Lower is better. Squaring makes big misses count extra.
    """
    squared_errors = [(a - p) ** 2 for a, p in zip(actual, predicted)]
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def mae(actual, predicted):
    """
    Mean Absolute Error: the average mistake size, without squaring.
    Easier to interpret than RMSE and less sensitive to a few big misses.
    """
    errors = [abs(a - p) for a, p in zip(actual, predicted)]
    return sum(errors) / len(errors)


def directional_accuracy(actual, predicted):
    """
    How often did the forecast get the DIRECTION right (up vs down)?

    For trading, direction often matters more than the exact number: if you
    know the ratio will rise you can make money even if you're off on by how
    much. Returns a fraction between 0 and 1.

    We compare, at each step, whether the predicted change and the actual change
    have the same sign.
    """
    correct = 0
    total = 0
    for t in range(1, len(actual)):
        actual_move = actual[t] - actual[t - 1]
        predicted_move = predicted[t] - actual[t - 1]
        if actual_move == 0:
            continue  # no real move to get right; skip it
        total += 1
        if (actual_move > 0) == (predicted_move > 0):
            correct += 1
    return correct / total if total else 0.0


def sharpe_ratio(profits):
    """
    Reward-to-risk of a strategy: average profit divided by how bumpy the
    profits are. Higher is better -- it means steady gains rather than a wild
    ride. Returns 0 if there's no variation to measure.
    """
    if not profits:
        return 0.0
    mean = sum(profits) / len(profits)
    variance = sum((p - mean) ** 2 for p in profits) / len(profits)
    std = math.sqrt(variance)
    return mean / std if std > 0 else 0.0
