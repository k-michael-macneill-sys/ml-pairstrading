"""
data.py
-------
Everything about getting the numbers ready.

The whole project is built around one idea: take two stocks whose prices tend
to move together, and watch the RATIO of their prices. If the ratio drifts away
from its usual level, we bet it will drift back.

This file loads a CSV of two stock prices and turns it into:
  1. the price ratio (a single list of numbers over time), and
  2. "windows": short slices of the past that we feed to a forecaster.
"""

import csv


def load_prices(csv_path):
    """
    Read a CSV with three columns: a timestamp, price A, and price B.

    Returns two lists of floats: prices_a and prices_b.

    Example row in the file:
        2023 01 03 10 09 00.000, 13.15, 14.75
    We ignore the timestamp and just keep the two prices.
    """
    prices_a = []
    prices_b = []

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip the header row (mask, A, B)
        for row in reader:
            if len(row) < 3:
                continue  # skip blank or broken rows
            prices_a.append(float(row[1]))
            prices_b.append(float(row[2]))

    return prices_a, prices_b


def compute_ratio(prices_a, prices_b):
    """
    Turn two price lists into one ratio list: ratio[t] = A[t] / B[t].

    The ratio is the only thing our models and strategies ever look at.
    """
    return [a / b for a, b in zip(prices_a, prices_b)]


def train_test_split(series, train_fraction=0.7):
    """
    Split a list into an earlier "training" part and a later "test" part.

    For time series you must NOT shuffle: the model should only ever learn from
    the past and be tested on the future. So we cut the list at one point.

    train_fraction=0.7 means the first 70% is for training, the last 30% is for
    testing.
    """
    cut = int(len(series) * train_fraction)
    return series[:cut], series[cut:]


def make_windows(series, window_size):
    """
    Slide a fixed-size window over the series to build (input, target) pairs.

    With window_size = 3 and series = [10, 11, 12, 13, 14]:

        inputs                target
        [10, 11, 12]     ->   13
        [11, 12, 13]     ->   14

    'inputs' is a list of windows (each window is a list of past values).
    'targets' is the single value that came right after each window.

    This is the standard way to frame "predict the next value" as something a
    model can learn from.
    """
    inputs = []
    targets = []
    for i in range(len(series) - window_size):
        inputs.append(series[i:i + window_size])
        targets.append(series[i + window_size])
    return inputs, targets
