#!/usr/bin/env python3

'''
Пример практической работы 1: регрессия средствами scikit-learn

Задание: 
A. Сгенерировать выборку данных, состоящую из элементов (трёхмерных точек) вида
    (x, 0.1 x + y, x^3 + y^3 + epsilon),
где x --- случайное число, равномерно распределённое в диапазоне [-1, 1),
y --- случайное число, нормально распределённое со средним значением 0.2
    и среднеквадратическим отклонением 1.3,
epsilon --- случайное число, нормально распределённое со средним значением 0
    и среднеквадратическим отклонением 0.1

Б. Вывести эту выборку в виде трёхмерного облака точек

В. Разделить выборку на обучающую и тестовую

Г. Провести регрессию на обучающей выборке методом Elastic Net
    (сочетанием регуляризаций ридж-регрессии и лассо-регрессии)
    средствами библиотеки scikit-learn

Д. Вывести тестовую выборку и предсказания для неё на один график
   для визуальной оценки результатов

Е. Оценить ошибку данного метода

'''

# импортируем используемые библиотеки:
import numpy  # базовая библиотека для численных методов
import sklearn.linear_model  # библиотека линейных моделей (в т.ч. линейной регрессии)
import matplotlib.pyplot as plt  # библиотека для вывода данных
from mpl_toolkits.mplot3d import axes3d, Axes3D  # дополнительные библиотеки
                                                 # для 3D визуализации

def generate_samples(NumSamples = 100):
    ''' функция генерации требуемой выборки средствами numpy.random:
    https://numpy.org/doc/stable/reference/random/index.html;
    NumSamples --- её размер ''' 
    # создаём матрицу случайных действительных чисел для первого параметра (x)
    # размером (NumSamples строк x 1 столбец), 
    # равномерно распределённых в диапазоне [-1, 1):
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
    v1 = numpy.random.uniform(low=-1, high=1, size=(NumSamples, 1))
    # создаём матрицу того же размера для второго параметра:
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
    v2 = v1 + numpy.random.normal(loc=0.2, scale=1.3, size=(NumSamples, 1))
    # создаём матрицу того же размера для правильного ответа:
    epsilon = numpy.random.normal(loc=0, scale=0.1, size=(NumSamples, 1))
    Y = numpy.power(v1, 3) + numpy.power(v2, 3) + epsilon
    # комбинируем первый и второй столбцы в одну матрицу, чтобы 
    # в каждой её строке были все независимые переменные
    X = numpy.concatenate((v1, v2), axis=1)
    return (X, Y)

def draw_samples(X, Y):
    '''Функция для вывода выборки
    X --- матрица независимых переменных,
    Y --- вектор зависимой переменной '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:,0], X[:,1], Y, label='samples')
    ax.set_xlabel('x')
    ax.set_ylabel('0.1x + y')
    ax.set_zlabel('x^3 + y^3 + epsilon')
    plt.show()

def split_dataset(X, Y, proportion_train=0.8):
    '''Разделить исходную выборку на обучающую и тестовую части;
    в принципе для этого имеется готовая функция:
    from sklearn.model_selection import train_test_split
    , но для иллюстрации процесса деление проделано вручную'''
    # размер обучающей выборки
    train_ds_sz = int(len(Y) * proportion_train)
    # перемешиваем номера всех элементов выборки
    dset_inds = list(range(len(Y)))
    numpy.random.shuffle(dset_inds)
    # начальная часть результата --- индексы обучающей выборки
    train_inds = dset_inds[:train_ds_sz]
    # конечная часть результата --- индексы тестовой выборки
    test_inds = dset_inds[train_ds_sz:]
    # выдаём части выборки
    return X[train_inds,:], Y[train_inds,:], X[test_inds,:], Y[test_inds,:]

def train_regression(X, Y):
    ''' Функция расчёта регрессионной модели встроенными средствами
    библиотеки scikit learn:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
    Модель Elastic Net минимизирует
    значение 1 / (2 * n_samples) * ||y - Xw||^2_2
    + alpha * l1_ratio * ||w||_1
    + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2,
    т.е взвешенную сумму трёх компонентов:
    1) квадратов уклонений предсказаний от результатов (как в обычной линейной регрессии);
    2) L1-нормы вектора параметров (близко по смыслу к лассо-регрессии, хотя и не то же самое);
    3) L2-нормы вектора параметров (как в ридж-регрессии).
    '''
    # создаём объект модели, передавая в него параметры регуляризации
    model = sklearn.linear_model.ElasticNet(alpha = 1., l1_ratio = 0.5)
    # обучаем модель на выборке встроенным в неё методом fit
    model.fit(X, Y)
    # выдаём модель
    return model

def draw_samples_and_prediction(X, Y, Y_pred):
    '''Функция для вывода выборки вместе с предсказаниями
    X --- матрица независимых переменных,
    Y --- вектор зависимой переменной,
    Y_pred --- вектор значений этой зависимой переменной,
       предсказанной по X '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:,0], X[:,1], Y, label='samples', marker='o')
    ax.scatter(X[:,0], X[:,1], Y_pred, label='prediction', marker='^')
    ax.set_xlabel('x')
    ax.set_ylabel('0.1x + y')
    ax.set_zlabel('x^3 + y^3 + epsilon')
    plt.show()

def calc_error(ground_truth, prediction):
    '''Функция для расчёта ошибки предсказания (среднего квадрата
    отклонения предсказанного значения функции от реального).
    ground_truth и prediction --- векторы-столбцы одного размера
    '''
    return numpy.mean(numpy.square(ground_truth - prediction))

# сгенерируем выборку в виде двух матриц, чтобы предсказываемое значение
# находилось в отдельной переменной

# зададим фиксированное начальное состояние генератора случайных чисел,
# чтобы можно было воспроизвести результаты (т.е. в выборку при каждом 
# запуске будут входить одни и те же псевдослучайные числа).
numpy.random.seed(17001)

# создаём выборку
X, Y = generate_samples(100)

# выводим её для визуальной оценки
draw_samples(X, Y)

# делим её на обучающую и тестовую части
X_train,Y_train,X_test,Y_test = split_dataset(X,Y)

# обучаем модель
model = train_regression(X_train, Y_train)

# предсказываем по ней результаты на тестовой выборке
Y_test_predicted = model.predict(X_test)

# выводим предсказания для тестовой выборки
draw_samples(X_test, Y_test_predicted)

# выводим одновременно предсказания и исходные данные
# для всей выборки в целом
Y_full_predicted = model.predict(X)
draw_samples_and_prediction(X, Y, Y_full_predicted)

# выводим ошибки предсказания:
error_train = calc_error(Y_train, model.predict(X_train))
print('Error on train dataset: {:.3f}'.format(error_train))
error_test = calc_error(Y_test, Y_test_predicted)
print('Error on test dataset: {:.3f}'.format(error_test))
error_full = calc_error(Y, Y_full_predicted)
print('Error on full dataset: {:.3f}'.format(error_full))
