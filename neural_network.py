from typing import Callable
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

ALPHA = 0.01
ITERATIONS = 10000
TEST_SIZE = 500
INPUT_LAYER_SIZE = 5
FIRST_LAYER_SIZE = 100
FIRST_LAYER_ACTIVATION = "relu"
SECOND_LAYER_SIZE = 100
SECOND_LAYER_ACTIVATION = "relu"
THIRD_LAYER_SIZE = 100
THIRD_LAYER_ACTIVATION = "relu"
OUTPUT_LAYER_SIZE = 4
OUTPUT_LAYER_ACTIVATION = "softmax"
TIMES = 10


def main():
    vec = range(TIMES)
    accuracies_train = []
    accuracies_test = []
    for i in vec:
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = read_data()
        w1, b1, w2, b2, w3, b3, wf, bf = gradient_descent(
            X_TRAIN, Y_TRAIN, ALPHA, ITERATIONS
        )
        if w1 is None:
            continue
        accuracy_train = test_gradient_descent(
            w1, b1, w2, b2, w3, b3, wf, bf, X_TRAIN, Y_TRAIN
        )
        accuracy_test = test_gradient_descent(
            w1, b1, w2, b2, w3, b3, wf, bf, X_TEST, Y_TEST
        )
        print(f"Accuracy (Train {i}): {(100*accuracy_train):.2f}%")
        print(f"Accuracy (Test {i}): {(100*accuracy_test):.2f}%")
        accuracies_train.append(accuracy_train)
        accuracies_test.append(accuracy_test)
    if len(accuracies_train) == 0:
        print("Failed to converge in all the times")
        return
    print(f"Accuracy mean: {100*np.mean(accuracies_train):.2f}%")
    print(f"Accuracy mean: {100*np.mean(accuracies_test):.2f}%")
    print(f"Accuracy Max(Train): {100*np.max(accuracies_train):.2f}%")
    print(f"Accuracy Max(Test): {100*np.max(accuracies_test):.2f}%")
    print(f"Accuracy Min(Train): {100*np.min(accuracies_train):.2f}%")
    print(f"Accuracy Min(Test): {100*np.min(accuracies_test):.2f}%")


def gradient_descent(
    x: np.ndarray, y: np.ndarray, alpha: float, iterations: int
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    w1, b1, w2, b2, w3, b3, wf, bf = init_params()
    for i in range(iterations):
        z1, a1, z2, a2, z3, a3, zf, af = forward_prop(w1, b1, w2, b2, w3, b3, wf, bf, x)
        dw1, db1, dw2, db2, dw3, db3, dwf, dbf = backward_prop(
            z1, a1, z2, a2, z3, a3, zf, af, w1, w2, w3, wf, x, y
        )
        w1, b1, w2, b2, w3, b3, wf, bf = update_params(
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            wf,
            bf,
            dw1,
            db1,
            dw2,
            db2,
            dw3,
            db3,
            dwf,
            dbf,
            alpha,
        )
        if i % 100 == 0:
            accuracy_train = test_gradient_descent(w1, b1, w2, b2, w3, b3, wf, bf, x, y)
            print(f"Accuracy (Train: {i}): {(100*accuracy_train):.2f}%")
            if i == 100 and accuracy_train < 0.5:
                print("Failed to converge")
                return None, None, None, None, None, None, None, None
    return w1, b1, w2, b2, w3, b3, wf, bf


def test_gradient_descent(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: np.ndarray,
    wf: np.ndarray,
    bf: np.ndarray,
    X_TEST: np.ndarray,
    Y_TEST: np.ndarray,
):
    af = forward_prop(w1, b1, w2, b2, w3, b3, wf, bf, X_TEST)[7]
    predictions = get_predictions(af)
    accuracy = get_accuracy(predictions, Y_TEST)
    return accuracy


def forward_prop(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: np.ndarray,
    wf: np.ndarray,
    bf: np.ndarray,
    x: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    l1_af, l1_deriv = get_activation(FIRST_LAYER_ACTIVATION)
    l2_af, l2_deriv = get_activation(SECOND_LAYER_ACTIVATION)
    l3_af, l3_deriv = get_activation(THIRD_LAYER_ACTIVATION)
    z1 = w1.dot(x) + b1
    a1 = l1_af(z1)
    z2 = w2.dot(a1) + b2
    a2 = l2_af(z2)
    z3 = w3.dot(a2) + b3
    a3 = l3_af(z3)
    zf = wf.dot(a3) + bf
    af = softmax(zf)
    return z1, a1, z2, a2, z3, a3, zf, af


def backward_prop(
    z1: np.ndarray,
    a1: np.ndarray,
    z2: np.ndarray,
    a2: np.ndarray,
    z3: np.ndarray,
    a3: np.ndarray,
    zf: np.ndarray,
    af: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    w3: np.ndarray,
    wf: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    l1_af, l1_deriv = get_activation(FIRST_LAYER_ACTIVATION)
    l2_af, l2_deriv = get_activation(SECOND_LAYER_ACTIVATION)
    l3_af, l3_deriv = get_activation(THIRD_LAYER_ACTIVATION)
    m = y.size
    one_hot_y = one_hot(y)
    dzf = af - one_hot_y
    dwf = 1 / m * dzf.dot(a3.T)
    dbf = 1 / m * np.sum(dzf)
    dz3 = wf.T.dot(dzf) * l3_deriv(z3)
    dw3 = 1 / m * dz3.dot(a2.T)
    db3 = 1 / m * np.sum(dz3)
    dz2 = w3.T.dot(dz3) * l2_deriv(z2)
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * l1_deriv(z1)
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1)
    return dw1, db1, dw2, db2, dw3, db3, dwf, dbf


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


def init_params() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    w1 = np.random.rand(FIRST_LAYER_SIZE, INPUT_LAYER_SIZE) - 0.5
    b1 = np.random.rand(FIRST_LAYER_SIZE, 1) - 0.5
    w2 = np.random.rand(SECOND_LAYER_SIZE, FIRST_LAYER_SIZE) - 0.5
    b2 = np.random.rand(SECOND_LAYER_SIZE, 1) - 0.5
    w3 = np.random.rand(THIRD_LAYER_SIZE, SECOND_LAYER_SIZE) - 0.5
    b3 = np.random.rand(THIRD_LAYER_SIZE, 1) - 0.5
    wf = np.random.rand(OUTPUT_LAYER_SIZE, THIRD_LAYER_SIZE) - 0.5
    bf = np.random.rand(OUTPUT_LAYER_SIZE, 1) - 0.5
    return w1, b1, w2, b2, w3, b3, wf, bf


def update_params(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: np.ndarray,
    wf: np.ndarray,
    bf: np.ndarray,
    dw1: np.ndarray,
    db1: np.ndarray,
    dw2: np.ndarray,
    db2: np.ndarray,
    dw3: np.ndarray,
    db3: np.ndarray,
    dwf: np.ndarray,
    dbf: np.ndarray,
    alpha: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    w3 = w3 - alpha * dw3
    b3 = b3 - alpha * db3
    wf = wf - alpha * dwf
    bf = bf - alpha * dbf
    return w1, b1, w2, b2, w3, b3, wf, bf


# ELEMENTARY FUNCTIONS:


def get_activation(
    activation: str,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    if activation == "relu":
        return relu, relu_deriv
    elif activation == "sigmoid":
        return sigmoid, sigmoid_deriv
    else:
        raise ValueError(f"Invalid activation function: {activation}")


def one_hot(y: np.ndarray) -> np.ndarray:
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


def softmax(z: np.ndarray) -> np.ndarray:
    z_shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_shifted)
    a = exp_z / np.sum(exp_z, axis=0, keepdims=True)
    return a


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0)


def relu_deriv(z: np.ndarray) -> np.ndarray:
    return np.float32(z > 0)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def sigmoid_deriv(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1 - s)


def get_predictions(a2: np.ndarray) -> np.ndarray:
    return np.argmax(a2, 0)


def get_accuracy(predictions: np.ndarray, y: np.ndarray) -> float:
    return np.sum(predictions == y) / y.size


if __name__ == "__main__":
    main()
