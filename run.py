"""
run.py
------
The main program. It wires the other files together and prints a report.

The pipeline, start to finish:
  1. Load two stock prices from a CSV and compute their ratio.
  2. Split the ratio into a training part (the past) and a test part (the future).
  3. Turn the training data into (window -> next value) pairs and fit a forecaster.
  4. Walk through the test data, predicting each next value from its window.
  5. Score the forecast, then run trading strategies on it and score those too.

Run it from the command line, for example:
    python run.py --data sample_data.csv --model linear --window 10

Use --help to see every option.
"""

import argparse

import data
import forecasters
import metrics
import plot
import strategies


def build_predictions(forecaster, ratios, window_size):
    """
    Produce a prediction for each point in `ratios` using the given forecaster.

    For the first `window_size` points we have no full window to look back on,
    so we just repeat the actual value (a harmless placeholder). From then on we
    feed the real preceding window to the forecaster.

    The returned list lines up index-for-index with `ratios`, which keeps the
    strategy code that consumes it easy to read.
    """
    predictions = []
    for t in range(len(ratios)):
        if t < window_size:
            predictions.append(ratios[t])
        else:
            window = ratios[t - window_size:t]
            predictions.append(forecaster.predict(window))
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Simple pairs-trading demo.")
    parser.add_argument("--data", required=True,
                        help="Path to the CSV file of two stock prices.")
    parser.add_argument("--model", default="linear",
                        choices=list(forecasters.FORECASTERS.keys()),
                        help="Which forecaster to use.")
    parser.add_argument("--window", type=int, default=10,
                        help="How many past values the forecaster looks at.")
    parser.add_argument("--train-fraction", type=float, default=0.7,
                        help="Fraction of data used for training (rest is test).")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Minimum signal size before a strategy will trade.")
    parser.add_argument("--plot", action="store_true",
                        help="Show a chart of the true vs predicted ratio.")
    args = parser.parse_args()

    # --- Step 1 & 2: load, compute the ratio, split into train and test ---
    prices_a, prices_b = data.load_prices(args.data)
    ratios = data.compute_ratio(prices_a, prices_b)
    train_ratios, test_ratios = data.train_test_split(ratios, args.train_fraction)

    print(f"Loaded {len(ratios)} rows "
          f"({len(train_ratios)} train, {len(test_ratios)} test).")

    # --- Step 3: build training windows and fit the forecaster ---
    train_inputs, train_targets = data.make_windows(train_ratios, args.window)
    forecaster = forecasters.FORECASTERS[args.model]()
    forecaster.fit(train_inputs, train_targets)
    print(f"Trained '{args.model}' forecaster on {len(train_inputs)} windows.\n")

    # --- Step 4: predict across the test period ---
    predictions = build_predictions(forecaster, test_ratios, args.window)

    # --- Step 5a: score the forecast itself ---
    # Skip the placeholder region at the start where we had no real window.
    scored_actual = test_ratios[args.window:]
    scored_pred = predictions[args.window:]

    if len(scored_actual) < 2:
        print("Not enough test data to score. Try a bigger CSV or a smaller "
              "--window / --train-fraction.")
        return

    print("--- Forecast quality ---")
    print(f"RMSE:                 {metrics.rmse(scored_actual, scored_pred):.6f}")
    print(f"MAE:                  {metrics.mae(scored_actual, scored_pred):.6f}")
    print(f"Directional accuracy: "
          f"{metrics.directional_accuracy(scored_actual, scored_pred):.1%}\n")

    # --- Step 5b: run each strategy on the predictions and score it ---
    print("--- Trading strategies ---")
    header = f"{'strategy':<14}{'total return':>14}{'trades':>9}{'win rate':>11}{'sharpe':>10}"
    print(header)
    print("-" * len(header))
    for name, strategy_fn in strategies.STRATEGIES.items():
        result = strategies.run_strategy(
            strategy_fn, test_ratios, predictions, args.threshold)
        sharpe = metrics.sharpe_ratio(result["profits"])
        print(f"{name:<14}"
              f"{result['total_profit']:>13.4%}"
              f"{result['num_trades']:>9}"
              f"{result['win_rate']:>10.1%}"
              f"{sharpe:>10.3f}")

    # --- Optional: draw the true vs predicted ratio ---
    if args.plot:
        plot.plot_true_vs_predicted(
            scored_actual, scored_pred, args.model,
            save_path="forecast_plot.png")


if __name__ == "__main__":
    main()
