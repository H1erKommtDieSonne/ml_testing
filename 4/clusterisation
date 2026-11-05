from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import davies_bouldin_score, mutual_info_score, adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances

base = Path(__file__).resolve().parent
output = base / "results"
output.mkdir(exist_ok=True)
rows = []

def evaluate(data, true_labels, labels, name, parameter, seed, iterations):
    k = len(np.unique(labels))
    db = davies_bouldin_score(data, labels) if 1 < k < len(data) else float("nan")
    mi = mutual_info_score(true_labels, labels)
    rows.append([name, parameter, seed, k, db, mi, iterations])
    print(name, parameter, seed, k, round(db, 4), round(mi, 4))

rng = np.random.default_rng(7)
a, b, e = rng.uniform(-1, 1, (3, 150))
first = np.column_stack((a, b, 0.5 + a + b + e))
a, b, e = rng.uniform(-1, 1, (3, 150))
second = np.column_stack((a, b + 0.5, a * e))
points = np.vstack((first, second))
true_labels = np.repeat([0, 1], 150)
order = rng.permutation(len(points))
points, true_labels = points[order], true_labels[order]
np.savetxt(output / "points.csv", np.column_stack((points, true_labels)), delimiter=",", header="x,y,z,group", comments="")
similarity = -euclidean_distances(points, squared=True)
stability = []
for quantile in [10, 50, 90]:
    preference = np.percentile(similarity[np.triu_indices(len(points), 1)], quantile)
    previous = None
    for seed in [0, 7, 42]:
        model = AffinityPropagation(preference=preference, damping=0.8, max_iter=1000, random_state=seed)
        labels = model.fit_predict(points)
        evaluate(points, true_labels, labels, "points_AP", quantile, seed, model.n_iter_)
        if previous is not None:
            stability.append([quantile, seed, adjusted_rand_score(previous, labels)])
        previous = labels.copy()
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(*points.T, c=labels, cmap="tab20", s=12)
        ax.set(xlabel="x", ylabel="y", zlabel="z", title=f"AP: q={quantile}, seed={seed}")
        fig.savefig(output / f"points_{quantile}_{seed}.png", dpi=150)
        plt.close(fig)
np.savetxt(output / "stability.csv", stability, delimiter=",", header="quantile,seed,ARI_previous_seed", comments="")

input_files = sorted((base / "texts" / "plants").glob("*.txt")) + sorted((base / "texts" / "animals").glob("*.txt"))
text_labels = np.repeat([0, 1], 10)
assert len(input_files) == 20
improved = base / "texts_improved"
improved.mkdir(exist_ok=True)
improved_files = []
for i, file in enumerate(input_files):
    text = file.read_text(encoding="utf-8")
    addition = " plant leaf root flower seed" if i < 10 else " animal body food habitat offspring"
    target = improved / file.name
    target.write_text(text + addition * 15, encoding="utf-8")
    improved_files.append(target)
for stage, files in [("original", input_files), ("improved", improved_files)]:
    count_vect = CountVectorizer(input="filename", encoding="utf-8")
    X_train_counts = count_vect.fit_transform([str(file) for file in files])
    print(stage, X_train_counts.shape)
    data = X_train_counts.toarray()
    for name, model in [("AP", AffinityPropagation(damping=0.8, max_iter=1000, random_state=7)), ("KMeans", KMeans(n_clusters=2, n_init=10, random_state=7))]:
        result = model.fit(data)
        print(result.labels_)
        evaluate(data, text_labels, result.labels_, stage + "_" + name, "default", 7, model.n_iter_)
        with (output / f"{stage}_{name}_labels.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["file", "group", "cluster"])
            writer.writerows(zip([file.name for file in files], text_labels, result.labels_))
with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["experiment", "quantile", "seed", "clusters", "Davies_Bouldin", "mutual_information", "iterations"])
    writer.writerows(rows)
