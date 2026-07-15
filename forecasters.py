"""
forecasters.py
--------------
Models that try to predict the NEXT value of the ratio from a window of recent
values.

Every forecaster has the same two methods so they're interchangeable:

    forecaster.fit(inputs, targets)   # learn from training data (some ignore it)
    forecaster.predict(window)        # given one window, guess the next value

We provide three, from simplest to smartest:
  1. LastValue        - "tomorrow looks like today" (a baseline to beat)
  2. MovingAverage    - average of the recent window
  3. LinearRegression - real machine learning, written from scratch with lists

Having a dumb baseline (LastValue) matters: if a fancy model can't beat "just
guess today's value again," the fancy model isn't actually learning anything
useful.
"""


class LastValue:
    """Predict that the next value equals the most recent value in the window."""

    def fit(self, inputs, targets):
        pass  # nothing to learn

    def predict(self, window):
        return window[-1]


class MovingAverage:
    """Predict the average of the last k values in the window."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, inputs, targets):
        pass  # nothing to learn

    def predict(self, window):
        recent = window[-self.k:]
        return sum(recent) / len(recent)


class LinearRegression:
    """
    Predict the next value as a weighted sum of the window values plus a bias:

        prediction = w[0]*x[0] + w[1]*x[1] + ... + w[n-1]*x[n-1] + bias

    The model LEARNS the weights by gradient descent: it starts with all
    weights at zero, makes predictions, sees how wrong it was, and nudges the
    weights in the direction that reduces the error. Repeat many times.

    This is the same core idea behind much larger neural networks, just with a
    single layer and no fancy activation functions.
    """

    def __init__(self, learning_rate=0.01, epochs=200):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def _predict_one(self, window):
        total = self.bias
        for w, x in zip(self.weights, window):
            total += w * x
        return total

    def fit(self, inputs, targets):
        n_features = len(inputs[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        n = len(inputs)

        for _ in range(self.epochs):
            # Accumulate the gradient (the direction of steepest error) over all
            # training examples, then take one step. This is "batch" gradient
            # descent -- simple and easy to follow.
            weight_gradient = [0.0] * n_features
            bias_gradient = 0.0

            for window, target in zip(inputs, targets):
                error = self._predict_one(window) - target
                for j in range(n_features):
                    weight_gradient[j] += error * window[j]
                bias_gradient += error

            # Average the gradient and step "downhill" against it.
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * (weight_gradient[j] / n)
            self.bias -= self.learning_rate * (bias_gradient / n)

    def predict(self, window):
        return self._predict_one(window)


# A lookup table so the command-line tool can pick a model by name.
FORECASTERS = {
    "last": LastValue,
    "average": MovingAverage,
    "linear": LinearRegression,
}
