"""
plot.py
-------
One job: draw the true ratio against the forecaster's prediction so you can SEE
how well the model is doing, instead of only reading error numbers.

We use matplotlib, the standard Python plotting library. If you don't have it:
    pip install matplotlib
"""

import matplotlib.pyplot as plt


def plot_true_vs_predicted(actual, predicted, model_name, save_path=None):
    """
    Draw two lines on the same chart: the real ratio and the predicted ratio.

    Where the lines sit on top of each other, the model is doing well. Where
    they separate, it's making mistakes. A good forecast "hugs" the true line.

    Inputs:
      actual     - the true ratio values over the test period
      predicted  - the model's guess for each of those same points
      model_name - just used for the title (e.g. "linear")
      save_path  - if given, the chart is also saved to this file (e.g. a PNG)
    """
    plt.figure(figsize=(11, 5))

    plt.plot(actual, color="steelblue", linewidth=1.2, label="True ratio")
    plt.plot(predicted, color="darkorange", linewidth=1.0,
             linestyle="--", label="Predicted ratio")

    plt.title(f"True vs Predicted Ratio ({model_name} model)")
    plt.xlabel("Time step (test period)")
    plt.ylabel("Price ratio (A / B)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved chart to {save_path}")

    plt.show()
