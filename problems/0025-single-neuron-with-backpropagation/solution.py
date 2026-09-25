import math
import numpy as np
def train_neuron(features, labels, weights, bias, learning_rate, epochs):
    mse_values = []

    for epoch in range(epochs):

        probabilities = []

        for x in features:
            z = sum(w * f for w, f in zip(weights, x)) + bias
            probability = 1 / (1 + math.exp(-z))
            probabilities.append(probability)

        mse = sum(
            (p - y) ** 2
            for p, y in zip(probabilities, labels)
        ) / len(labels)

        mse_values.append(mse)

        # 3. Calculate gradients
        weight_gradients = [0.0] * len(weights)
        bias_gradient = 0.0

        for x, y, p in zip(features, labels, probabilities):

            dz = 2 * (p - y) * p * (1 - p)

            for j in range(len(weights)):
                weight_gradients[j] += dz * x[j]

            bias_gradient += dz

        n = len(features)

        weight_gradients = [
            gradient / n
            for gradient in weight_gradients
        ]

        bias_gradient /= n

        for j in range(len(weights)):
            weights[j] -= learning_rate * weight_gradients[j]

        bias -= learning_rate * bias_gradient

    weights = [round(w, 4) for w in weights]
    bias = round(bias, 4)
    mse_values = [round(mse, 4) for mse in mse_values]

    return weights, bias, mse_values