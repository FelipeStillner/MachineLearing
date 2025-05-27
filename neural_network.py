import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

ALPHA = 0.001
ITERATIONS = 10000
TEST_SIZE = 500
INPUT_LAYER_SIZE = 5
HIDDEN_LAYER_SIZE = 5
OUTPUT_LAYER_SIZE = 4
TIMES = 10


def main():
    vec = range(TIMES)
    accuracies_train = []
    accuracies_test = []
    for i in vec:
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = read_data()
        w1, b1, w2, b2 = gradient_descent(X_TRAIN, Y_TRAIN, ALPHA, ITERATIONS)
        accuracy_train = test_gradient_descent(w1, b1, w2, b2, X_TRAIN, Y_TRAIN)
        accuracy_test = test_gradient_descent(w1, b1, w2, b2, X_TEST, Y_TEST)
        print(f"Accuracy (Train {i}): {(100*accuracy_train):.2f}%")
        print(f"Accuracy (Test {i}): {(100*accuracy_test):.2f}%")
        accuracies_train.append(accuracy_train)
        accuracies_test.append(accuracy_test)
    print(f"Accuracy mean: {100*np.mean(accuracies_train):.2f}%")
    print(f"Accuracy mean: {100*np.mean(accuracies_test):.2f}%")
    print(f"Accuracy Max(Train): {100*np.max(accuracies_train):.2f}%")
    print(f"Accuracy Max(Test): {100*np.max(accuracies_test):.2f}%")
    print(f"Accuracy Min(Train): {100*np.min(accuracies_train):.2f}%")
    print(f"Accuracy Min(Test): {100*np.min(accuracies_test):.2f}%")


def gradient_descent(
    x: np.ndarray, y: np.ndarray, alpha: float, iterations: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = backward_prop(z1, a1, z2, a2, w1, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        # if i % 10 == 0:
        #     accuracy = get_accuracy(get_predictions(a2), y)
        #     print(f"Accuracy (Train: {i}): {(100*accuracy):.2f}%")
    return w1, b1, w2, b2


def test_gradient_descent(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    X_TEST: np.ndarray,
    Y_TEST: np.ndarray,
):
    a2 = forward_prop(w1, b1, w2, b2, X_TEST)[3]
    predictions = get_predictions(a2)
    accuracy = get_accuracy(predictions, Y_TEST)
    return accuracy


def forward_prop(
    w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z1 = w1.dot(x) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2


def backward_prop(
    z1: np.ndarray,
    a1: np.ndarray,
    z2: np.ndarray,
    a2: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = y.size
    one_hot_y = one_hot(y)
    dz2 = a2 - one_hot_y
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * relu_deriv(z1)
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1)
    return dw1, db1, dw2, db2


def read_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = pd.read_csv("data.csv")
    data = np.array(data)
    np.random.shuffle(data)

    data_test = data[0:TEST_SIZE]
    data_train = data[TEST_SIZE:]

    data_train = data_train.T
    y_train = (data_train[7]).astype(np.int64) - 1
    x_train = data_train[1:6]

    data_test = data_test.T
    y_test = data_test[7].astype(np.int64) - 1
    x_test = data_test[1:6]

    return x_train, y_train, x_test, y_test


def init_params() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w1 = np.random.rand(HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE) - 0.5
    b1 = np.random.rand(HIDDEN_LAYER_SIZE, 1) - 0.5
    w2 = np.random.rand(OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE) - 0.5
    b2 = np.random.rand(OUTPUT_LAYER_SIZE, 1) - 0.5
    return w1, b1, w2, b2


def update_params(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    dw1: np.ndarray,
    db1: np.ndarray,
    dw2: np.ndarray,
    db2: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2


# ELEMENTARY FUNCTIONS:


def one_hot(y: np.ndarray) -> np.ndarray:
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0)


def softmax(z: np.ndarray) -> np.ndarray:
    exp_z = np.exp(z)
    a = exp_z / np.sum(exp_z, axis=0, keepdims=True)
    return a


def relu_deriv(z: np.ndarray) -> np.ndarray:
    return np.float32(z > 0)


def get_predictions(a2: np.ndarray) -> np.ndarray:
    return np.argmax(a2, 0)


def get_accuracy(predictions: np.ndarray, y: np.ndarray) -> float:
    return np.sum(predictions == y) / y.size


if __name__ == "__main__":
    main()
