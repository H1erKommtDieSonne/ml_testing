from pathlib import Path
from time import perf_counter
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

out = Path(__file__).resolve().parent / 'results'
out.mkdir(exist_ok=True)
rng = np.random.default_rng(7)

def save_plot(name):
    plt.tight_layout()
    plt.savefig(out / name, dpi=140)
    plt.close()

def sign(x):
    return (x > 0) * 1.0

def logistic(x, s=1):
    return 1 / (1 + np.exp(-s * x))

class ManualMLP:
    def __init__(self, coeffs, intercepts, act_func, output_func=None):
        self.coeffs = coeffs
        self.intercepts = intercepts
        self.act_func = act_func
        self.output_func = output_func or act_func

    def predict(self, X):
        curr_val = X
        results = []
        for i in range(len(self.coeffs)):
            curr_val = np.matmul(curr_val, self.coeffs[i]) + self.intercepts[i]
            results.append(curr_val)
            func = self.output_func if i == len(self.coeffs) - 1 else self.act_func
            curr_val = func(curr_val)
            results.append(curr_val)
        return curr_val, results

def draw_all_layers_output(net, name):
    px = np.linspace(-4, 4, 161)
    py = np.linspace(-2, 2, 81)
    vx, vy = np.meshgrid(px, py)
    test_set = np.column_stack((vx.ravel(), vy.ravel()))
    preds, interm = net.predict(test_set)
    rows = max(values.shape[1] for values in interm)
    fig, axs = plt.subplots(rows, len(interm), figsize=(15, 2 * rows), squeeze=False)
    for i, values in enumerate(interm):
        for j in range(rows):
            ax = axs[j, i]
            if j < values.shape[1]:
                ax.imshow(values[:, j].reshape(vy.shape), origin='lower', extent=(-4, 4, -2, 2), cmap='coolwarm', aspect='auto')
                ax.set_title(f'L{i // 2 + 1}, N{j + 1}: ' + ('sum' if i % 2 == 0 else 'activation'))
            else:
                ax.axis('off')
    save_plot(name)

def quality(expected, predicted):
    rep = classification_report(expected, predicted, output_dict=True, zero_division=0)
    return rep['1']['precision'], rep['1']['recall']

layer1_c = np.array([[1, -1, 1, -1, 0, 0], [0, 0, 0, 0, 1, -1]])
layer1_i = np.array([3, -1, -1, 3, 1, 1])
layer2_c = np.array([[1, 0, -1, 0, 0, 0, 0], [1, 0, 0, -1, 0, 0, 0], [0, 1, 0, -1, 0, 0, 0], [0, 1, 0, 0, -1, 0, 0], [1, 1, 0, 0, 0, -1, 0], [1, 1, 0, 0, 0, 0, -1]])
layer2_i = np.array([-3.5, -3.5, 0.5, 0.5, 0.5, 0.5, 0.5])
layer3_c = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
layer3_i = np.array([-0.5, -0.5])
coeffs = [layer1_c, layer2_c, layer3_c]
intercepts = [layer1_i, layer2_i, layer3_i]

N = 10000
xs = rng.uniform(-4, 4, size=N)
ys = rng.uniform(-2, 2, size=N)
area1 = (xs > -3) & (xs < -1) & (ys > -1) & (ys < 1)
area2 = (xs > 1) & (xs < 3) & (ys > -1) & (ys < 1)
class_ids = (area1 | area2).astype(int)
points = np.column_stack((xs, ys))
points_train, points_test, classes_train, classes_test = train_test_split(points, class_ids, train_size=0.8, random_state=17, stratify=class_ids)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for ax, X, y, title in zip(axs, [points_train, points_test], [classes_train, classes_test], ['Обучение', 'Тест']):
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=3, vmin=0, vmax=1)
    ax.set(xlabel='x', ylabel='y', title=title, aspect='equal')
save_plot('01_samples_2d.png')

manual_rows = []
for name, func in [('step', sign), ('logistic10', lambda x: logistic(x, 10)), ('logistic1', logistic), ('relu', lambda x: np.maximum(x, 0))]:
    manual = ManualMLP(coeffs, intercepts, func)
    output, interm = manual.predict(points_test)
    predicted = (output[:, 0] > output[:, 1]).astype(int)
    p, r = quality(classes_test, predicted)
    manual_rows.append([name, p, r])
    draw_all_layers_output(manual, f'02_manual_{name}.png')
np.savetxt(out / 'manual.csv', manual_rows, fmt='%s', delimiter=';', header='activation;precision;recall', comments='')

layer1_sizes = [5, 6, 7]
layer2_sizes = [6, 7, 8]
precisions = np.zeros((3, 3))
recalls = np.zeros((3, 3))
NE = 3
best_score = -1
training_rows = []
for e_num in range(NE):
    for i, layer1_size in enumerate(layer1_sizes):
        for j, layer2_size in enumerate(layer2_sizes):
            mlp = MLPClassifier(hidden_layer_sizes=(layer1_size, layer2_size), activation='logistic', learning_rate_init=0.01, max_iter=1000, random_state=17 + e_num)
            mlp.fit(points_train, classes_train)
            p, r = quality(classes_test, mlp.predict(points_test))
            precisions[i, j] += p
            recalls[i, j] += r
            training_rows.append([e_num + 1, layer1_size, layer2_size, p, r, mlp.n_iter_])
            train_p, train_r = quality(classes_train, mlp.predict(points_train))
            score = 2 * train_p * train_r / (train_p + train_r + 1e-15)
            if layer1_size == 6 and layer2_size == 8 and score > best_score:
                best_score, selected = score, deepcopy(mlp)
            print(f'2D: repeat={e_num + 1}, layers=({layer1_size}, {layer2_size}), P={p:.4f}, R={r:.4f}', flush=True)
precisions /= NE
recalls /= NE
np.savetxt(out / 'networks_2d.csv', training_rows, delimiter=';', header='repeat;layer1;layer2;precision;recall;epochs', comments='')
for name, values in [('precision', precisions), ('recall', recalls)]:
    plt.figure(figsize=(6, 4))
    plt.imshow(values, origin='lower', extent=(5.5, 8.5, 4.5, 7.5), vmin=0, vmax=1)
    plt.xticks(layer2_sizes)
    plt.yticks(layer1_sizes)
    plt.xlabel('Второй скрытый слой')
    plt.ylabel('Первый скрытый слой')
    plt.title(name)
    plt.colorbar()
    for i in range(3):
        for j in range(3):
            plt.text(layer2_sizes[j], layer1_sizes[i], f'{values[i, j]:.3f}', ha='center', va='center')
    save_plot(f'03_{name}.png')

trained = ManualMLP(selected.coefs_, selected.intercepts_, logistic)
draw_all_layers_output(trained, '04_trained_layers.png')
np.savez(out / 'trained_weights.npz', **{f'W{i + 1}': w for i, w in enumerate(selected.coefs_)}, **{f'b{i + 1}': b for i, b in enumerate(selected.intercepts_)})
ablation_rows = []
for count in [0, 2, 4]:
    damaged = deepcopy(selected)
    damaged.coefs_[-2][:, :count] = 0
    damaged.intercepts_[-2][:count] = 0
    damaged.coefs_[-1][:count, :] = 0
    p, r = quality(classes_test, damaged.predict(points_test))
    ablation_rows.append([count, count / 8, p, r])
np.savetxt(out / 'ablation.csv', ablation_rows, delimiter=';', header='disabled;fraction;precision;recall', comments='')

M = 1500
cube = rng.uniform([0, -0.5, -0.5], [1, 0.5, 0.5], size=(M, 3))
xy = rng.uniform(-3, 3, size=(M, 2))
z = -3 - xy[:, 0] + xy[:, 1] - rng.uniform(0.1, 2, size=M)
below = np.column_stack((xy, z))
slab_parts = []
while sum(len(part) for part in slab_parts) < 2 * M:
    xy = rng.uniform(-3, 3, size=(2 * M, 2))
    z = rng.uniform(-1, 1, size=2 * M) - 0.2 * xy[:, 0] - 0.2 * xy[:, 1]
    part = np.column_stack((xy, z))
    in_cube = (part[:, 0] > 0) & (part[:, 0] < 1) & (abs(part[:, 1]) < 0.5) & (abs(part[:, 2]) < 0.5)
    in_below = part[:, 0] - part[:, 1] + part[:, 2] < -3
    slab_parts.append(part[~(in_cube | in_below)])
slab = np.vstack(slab_parts)[:2 * M]
points3 = np.vstack((cube, below, slab))
classes3 = np.concatenate((np.ones(2 * M, dtype=int), np.zeros(2 * M, dtype=int)))
X_train, X_test, y_train, y_test = train_test_split(points3, classes3, train_size=0.8, random_state=17, stratify=classes3)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
for values, label, color in [(cube, 'Тип 3, класс 1', 'red'), (below, 'Тип 5, класс 1', 'orange'), (slab, 'Тип 6, класс 0', 'blue')]:
    ax.scatter(values[:, 0], values[:, 1], values[:, 2], s=2, alpha=0.3, c=color, label=label)
ax.set(xlabel='x', ylabel='y', zlabel='z')
ax.legend()
save_plot('05_samples_3d.png')

def experiment3(layers):
    net = MLPClassifier(hidden_layer_sizes=layers, activation='relu', learning_rate_init=0.003, max_iter=800, random_state=17)
    start = perf_counter()
    net.fit(X_train, y_train)
    elapsed = perf_counter() - start
    train_p, train_r = quality(y_train, net.predict(X_train))
    test_p, test_r = quality(y_test, net.predict(X_test))
    print(f'3D: layers={layers}, time={elapsed:.2f}, epochs={net.n_iter_}, train=({train_p:.4f}, {train_r:.4f}), test=({test_p:.4f}, {test_r:.4f})', flush=True)
    return [elapsed, net.n_iter_, elapsed / net.n_iter_, train_p, train_r, test_p, test_r]

sizes = np.arange(10, 60, 5)
results3 = np.array([[size] + experiment3((int(size), 8)) for size in sizes])
np.savetxt(out / 'networks_3d.csv', results3, delimiter=';', header='layer1;seconds;epochs;seconds_per_epoch;train_precision;train_recall;test_precision;test_recall', comments='')
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for ax, col, title in zip(axs, [1, 3], ['Полное время обучения, с', 'Среднее время эпохи, с']):
    ax.plot(sizes, results3[:, col], 'o-')
    ax.set(xlabel='Нейроны первого скрытого слоя', ylabel=title)
    ax.grid()
save_plot('06_training_time.png')
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for ax, train_col, test_col, title in zip(axs, [4, 5], [6, 7], ['Precision', 'Recall']):
    ax.plot(sizes, results3[:, train_col], 'o-', label='Обучение')
    ax.plot(sizes, results3[:, test_col], 's-', label='Тест')
    ax.set(xlabel='Нейроны первого скрытого слоя', ylabel=title, ylim=(0, 1.02))
    ax.legend()
    ax.grid()
save_plot('07_quality_3d.png')
depth_results = np.array([[len(layers)] + experiment3(layers) for layers in [(20,), (20, 8), (20, 8, 8)]])
np.savetxt(out / 'depth_3d.csv', depth_results, delimiter=';', header='hidden_layers;seconds;epochs;seconds_per_epoch;train_precision;train_recall;test_precision;test_recall', comments='')
print('Результаты сохранены в', out)
