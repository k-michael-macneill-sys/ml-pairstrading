# Pairs Trading — A Beginner-Friendly Version

A small, readable project that predicts the price **ratio** of two related
stocks and tests simple trading strategies on the prediction. It's built to be
understood (and rewritten) by a first- or second-year computer science student:
no machine-learning libraries, no frameworks — just plain Python and a bit of
math you can follow line by line.

## The idea in one paragraph

Some pairs of stocks tend to move together (think two big banks). If you divide
one price by the other you get a **ratio** that usually hovers around some level.
When the ratio drifts away from that level, a "pairs trade" bets it will drift
back. This project (1) learns to forecast the next ratio value and (2) turns
those forecasts into buy/sell decisions, then measures how well each strategy
did.

## How the pieces fit together

```
CSV of two prices
      │
      ▼
  data.py        load prices, compute ratio, split into train/test, make windows
      │
      ▼
forecasters.py   learn to predict the next ratio (3 models to choose from)
      │
      ▼
strategies.py    turn predictions into +1 / -1 / 0 positions, compute profit
      │
      ▼
  metrics.py     score the forecasts and the strategies
      │
      ▼
   run.py        the main program: runs all of the above and prints a report
```

Each file does one job and can be read on its own in a few minutes.

## The three forecasters

| Name      | What it does                                   | Learns? |
|-----------|------------------------------------------------|---------|
| `last`    | Predicts the next value equals the last value  | No      |
| `average` | Predicts the average of the recent window      | No      |
| `linear`  | Weighted sum of the window, weights learned by gradient descent | Yes |

`last` is a **baseline**: any model worth using should beat "just guess today's
value again." Keeping a baseline around is a habit worth building early.

## The three strategies

- **reversion** — bet the recent move reverses (ignores the forecast).
- **forecasting** — bet in the direction the model predicts (ignores the recent move).
- **hybrid** — only trade when reversion and the forecast agree; fewer but
  better-supported trades.

## Running it

You need Python 3 (nothing else to install).

```bash
python run.py --data sample_data.csv --model linear --window 10
```

Options:

| Flag               | Meaning                                          | Default |
|--------------------|--------------------------------------------------|---------|
| `--data`           | Path to the CSV (required)                        | —       |
| `--model`          | `last`, `average`, or `linear`                    | linear  |
| `--window`         | How many past values the forecaster looks at      | 10      |
| `--train-fraction` | Fraction of data used for training                | 0.7     |
| `--threshold`      | Minimum signal size before a strategy trades      | 0.0     |
| `--plot`           | Show a chart of the true vs predicted ratio       | off     |

Add `--plot` to any run to pop up a chart of the true ratio versus the model's
prediction (it also saves a `forecast_plot.png`). Where the two lines sit on top
of each other, the model is doing well:

```bash
python run.py --data sample_data.csv --model average --window 10 --plot
```

The included `sample_data.csv` is only six rows — enough to see the format, too
small to produce real results. Point `--data` at a larger file to see the full
report (forecast quality plus a table of strategy returns, trade counts, win
rates, and Sharpe ratios).

## Data format

A CSV with a header row and three columns: a timestamp (ignored), then the two
prices. The ratio is computed as `price_A / price_B`.

```
mask,A,B
2023 01 03 10 09 00.000,13.15,14.75
2023 01 03 10 10 00.000,13.10,14.68
```

## Ideas to extend it (good exercises)

- Add transaction costs to `strategies.py` and watch profits shrink.
- Add a new forecaster (e.g. weighted moving average) — you only need `fit` and
  `predict` methods to plug into everything else.
- Try different `--window` sizes and see how forecast accuracy changes.
- Zoom the chart into a smaller slice (say 200 points) to see the model
  tracking short-term wiggles rather than the whole test period at once.

## A note on what this is (and isn't)

This is a **learning project**. The profit numbers assume you can trade instantly
with no fees or slippage, which is never true in real markets, so they will look
far rosier than anything achievable in practice. The goal is to understand the
mechanics clearly, not to trade real money.
