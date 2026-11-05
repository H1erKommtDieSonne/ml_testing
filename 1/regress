#!/usr/bin/env python3


# импортируем используемые библиотеки:
import numpy as np  # базовая библиотека для численных методов
import sklearn.linear_model  # библиотека линейных моделей (в т.ч. линейной регрессии)
import matplotlib.pyplot as plt  # библиотека для вывода данных
from mpl_toolkits.mplot3d import axes3d, Axes3D  # дополнительные библиотеки
                                                 # для 3D визуализации
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split



NUM_SAMPLES = 300
PROPORTION_TRAIN = 0.8

ALPHA = 1.0
BANDWIDTH = 1.0

SEED = 17007


def generate_samples(NumSamples=100):

    X = np.random.normal(
        loc=4.0,
        scale=3.0,
        size=(NumSamples, 1)
    )

    epsilon = np.random.normal(
        loc=0.0,
        scale=0.1,
        size=NumSamples
    )


    Y = 0.01 * X[:, 0] + epsilon

    return X, Y


def draw_samples(X, Y, title="Исходная выборка"):

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(
        X[:, 0],
        Y,
        s=25,
        alpha=0.7,
        label="Точки выборки"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()


def split_dataset(X, Y, proportion_train=0.8):

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        train_size=proportion_train,
        random_state=SEED,
        shuffle=True
    )

    return X_train, Y_train, X_test, Y_test


def train_regression(X, Y, alpha=1.0):

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, Y)

    return model


class GaussianKernelRegression:


    def __init__(self, bandwidth=1.0):


        self.bandwidth = bandwidth
        self.X_train = None
        self.Y_train = None

    def fit(self, X, Y):

        self.X_train = np.asarray(X[:, 0], dtype=float).copy()
        self.Y_train = np.asarray(Y, dtype=float).reshape(-1).copy()



        return self

    def predict(self, X):



        predictions = np.empty(len(X))

        for i, x in enumerate(X[:, 0]):

            distances = (x - self.X_train) / self.bandwidth


            log_weights = -0.5 * distances**2


            log_weights -= np.max(log_weights)

            weights = np.exp(log_weights)

            predictions[i] = (
                np.sum(weights * self.Y_train)
                / np.sum(weights)
            )

        return predictions


def draw_samples_and_prediction(X,Y,Y_pred,title="Предсказания"):

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(
        X[:, 0],
        Y,
        s=25,
        alpha=0.6,
        label="Реальные значения",
        marker="o"
    )

    ax.scatter(
        X[:, 0],
        Y_pred,
        s=35,
        label="Предсказания",
        marker="^"
    )


    order = np.argsort(X[:, 0])

    ax.plot(
        X[order, 0],
        np.asarray(Y_pred)[order],
        alpha=0.7,
        linewidth=1
    )


    ax.plot(
        X[order, 0],
        0.01 * X[order, 0],
        "k--",
        label="Истинная зависимость: y = 0.01x"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()


def calc_error(ground_truth, prediction):
    ground_truth = np.asarray(ground_truth).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)



    return float(np.mean((ground_truth - prediction)**2))


def main():

    np.random.seed(SEED)


    X, Y = generate_samples(NUM_SAMPLES)


    draw_samples(X, Y, "Исходная выборка")


    X_train, Y_train, X_test, Y_test = split_dataset(X,Y,proportion_train=PROPORTION_TRAIN)

    print(f"Размер исходной выборки: {len(Y)}")
    print(f"Размер обучающей части:  {len(Y_train)}")
    print(f"Размер тестовой части:   {len(Y_test)}")


    ridge_model = train_regression(X_train, Y_train, ALPHA)

    kernel_model = GaussianKernelRegression(BANDWIDTH)
    kernel_model.fit(X_train, Y_train)

    models = {
        "Ridge": ridge_model,
        "Гауссово сглаживание": kernel_model
    }



    X_new, Y_new = generate_samples(NUM_SAMPLES)

    datasets = {
        "Обучающая": (X_train, Y_train),
        "Тестовая": (X_test, Y_test),
        "Исходная": (X, Y),
        "Новая": (X_new, Y_new)
    }

    errors = {}

    for model_name, model in models.items():
        print(f"\n{model_name}")
        print(f"{'Выборка':<20} {'MSE':>12} {'RMSE':>12}")

        errors[model_name] = {}


        for dataset_name, (X_part, Y_part) in datasets.items():
            Y_pred = model.predict(X_part)

            mse = calc_error(Y_part, Y_pred)
            rmse = np.sqrt(mse)

            errors[model_name][dataset_name] = mse

            print(
                f"{dataset_name:<20} "
                f"{mse:>12.6f} "
                f"{rmse:>12.6f}"
            )


        draw_samples(
            X_test,
            model.predict(X_test),
            title=f"{model_name}: предсказания на тестовой части"
        )


        for dataset_name in ["Тестовая", "Исходная", "Новая"]:
            X_part, Y_part = datasets[dataset_name]

            draw_samples_and_prediction(
                X_part,
                Y_part,
                model.predict(X_part),
                title=f"{model_name}: {dataset_name.lower()} выборка"
            )







    plt.show()


if __name__ == "__main__":
    main()
