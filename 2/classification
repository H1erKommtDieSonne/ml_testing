

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


NUM_SAMPLES = 1000
PROPORTION_TRAIN = 0.7
SEED = 7007
N_COMPONENTS = 4
OUT = Path(__file__).resolve().parent / "results"
SCHEMES = {
    "A": ("Тип 2 против типов 7+10", ["тип 2", "типы 7+10"]),
    "B": ("Типы 2+7 против типа 10", ["типы 2+7", "тип 10"]),
    "C": ("Три отдельных класса", ["тип 2", "тип 7", "тип 10"]),
}


def generate_samples(n, seed):
    rng = np.random.default_rng(seed)

    # Тип 2 (a, b - 0.5, a * epsilon)
    a, b, eps = rng.uniform(-1, 1, size=(3, n))
    group2 = np.column_stack((a, b - 0.5, a * eps))

    # Тип 7 поверхность x² + y² + (z + 0.5)² = 1 при z <= 0
    u = rng.uniform(-1, 0.5, n)  # u = z + 0.5.
    phi = rng.uniform(0, 2 * np.pi, n)
    radius = np.sqrt(1 - u**2)
    group7 = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), u - 0.5))

    # Тип 10 (x, y, exp(x - y) + epsilon), СКО x и y равно 3
    x, y = rng.normal(loc=1, scale=3, size=(2, n))
    eps = rng.uniform(-0.1, 0.1, n)
    group10 = np.column_stack((x, y, np.exp(x - y) + eps))
    return np.vstack((group2, group7, group10)), np.repeat([2, 7, 10], n)


def make_labels(groups, scheme):
    if scheme == "A":
        return (groups != 2).astype(int)
    if scheme == "B":
        return (groups == 10).astype(int)
    return np.searchsorted([2, 7, 10], groups)


class GaussianMixtureClassifier:


    def fit(self, X, y):
        self.classes, counts = np.unique(y, return_counts=True)
        self.priors = counts / counts.sum()
        self.models = []
        for label in self.classes:
            model = GaussianMixture(
                n_components=N_COMPONENTS, covariance_type="full",
                reg_covar=1e-6, n_init=5, max_iter=1000,
                tol=1e-4, random_state=SEED + int(label),
            )
            model.fit(X[y == label])
            self.models.append(model)

    def predict(self, X):

        scores = []
        for model, prior in zip(self.models, self.priors):
            scores.append(model.score_samples(X) + np.log(prior))
        return self.classes[np.argmax(np.column_stack(scores), axis=1)]


def create_models():
    return {
        "Пороговый": SGDClassifier(
            loss="perceptron", penalty="l2", alpha=0.001,
            learning_rate="constant", eta0=0.01,
            max_iter=10000, tol=1e-5, n_iter_no_change=30,
            average=True, random_state=SEED,
        ),
        "SVM": LinearSVC(
            C=1.0, dual=False, max_iter=20000, tol=1e-5, random_state=SEED,
        ),
        "Гауссова смесь": GaussianMixtureClassifier(),
    }


def draw_data(X, groups):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(projection="3d")
    for group in [2, 7, 10]:
        points = X[groups == group]
        ax.scatter(points[:, 0], points[:, 1], np.arcsinh(points[:, 2]),
                   s=8, alpha=0.5, label=f"Тип {group}")
    ax.set(xlabel="x", ylabel="y", zlabel="asinh(z)", title="Исходные группы точек")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "groups.png", dpi=150)
    plt.close(fig)


def draw_confusions(matrices):
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    for ax, (title, names, matrix) in zip(axes.flat, matrices):
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > matrix.max() / 2 else "black")
        ax.set_xticks(range(len(names)), names, rotation=15)
        ax.set_yticks(range(len(names)), names)
        ax.set(title=title, xlabel="Предсказанный класс", ylabel="Истинный класс")
    fig.suptitle("Матрицы ошибок на отложенной выборке")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, groups = generate_samples(NUM_SAMPLES, SEED)
    X_new, groups_new = generate_samples(NUM_SAMPLES, SEED + 1)
    X_train, X_test, groups_train, groups_test = train_test_split(
        X, groups, train_size=PROPORTION_TRAIN, stratify=groups, random_state=SEED,
    )


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_original = scaler.transform(X)
    X_new = scaler.transform(X_new)

    report = ["Отчет", "",
              f"По {NUM_SAMPLES} точек каждого типа. Train: {len(X_train)}, test: {len(X_test)}.", "",
              "![Исходные группы](groups.png)", "",
              "На рисунке z сжата функцией asinh", ""]
    matrices = []
    comparison = []
    for scheme, (title, names) in SCHEMES.items():
        y_train = make_labels(groups_train, scheme)
        y_test = make_labels(groups_test, scheme)
        report += [f"## {scheme}: {title}", "",
                   "Метрики на отложенной выборке:", "",
                   "| Метод | Accuracy | Класс | Precision | Recall |",
                   "|---|---:|---|---:|---:|"]
        for name, model in create_models().items():
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            accuracy = accuracy_score(y_test, prediction)
            precision, recall, _, _ = precision_recall_fscore_support(
                y_test, prediction, labels=range(len(names)), zero_division=0,
            )
            for i, label in enumerate(names):
                report.append(f"| {name} | {accuracy:.4f} | {label} | {precision[i]:.4f} | {recall[i]:.4f} |")
            matrices.append((f"{scheme}: {name}", names, confusion_matrix(y_test, prediction)))
            print(f"{scheme}: {name}, accuracy = {accuracy:.4f}")

            if scheme == "A":
                old = accuracy_score(make_labels(groups, scheme), model.predict(X_original))
                new = accuracy_score(make_labels(groups_new, scheme), model.predict(X_new))
                comparison.append(f"| {name} | {old:.4f} | {accuracy:.4f} | {new:.4f} | {100*(new-accuracy):+.2f} |")
        report.append("")

    report += ["## Новая выборка: тип 2 против типов 7+10", "",
               "| Метод | Старая полная | Старый test | Новая | Новая минус test, п.п. |",
               "|---|---:|---:|---:|---:|", *comparison, "",
               "Старая полная выборка содержит обучающие точки. Для независимого сравнения используем test и новую выборку.", "",
               "![Матрицы ошибок](confusion_matrices.png)", ""]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    draw_data(X, groups)
    draw_confusions(matrices)
    print(f"Графики и краткий отчёт сохранены в {OUT}")


if __name__ == "__main__":
    main()
