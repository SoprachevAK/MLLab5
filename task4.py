import numpy as np
from pandas import read_csv
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

FILENAME = "../datasets/longley.csv"


def read_data(filename):
    file = read_csv(filename)
    target = file["Employed"]
    file.drop(["Employed", "Population"], axis=1, inplace=True)
    data = file
    return data, target


def main():
    data, target = read_data(FILENAME)
    array = []
    linear_clf = LinearRegression()
    mae_linear_test = []
    mae_linear_train = []
    for i in range(1000):
        array.append(i)
        train_data, test_data, train_target, test_target = train_test_split(data, target, train_size=0.5)
        linear_clf.fit(train_data, train_target)
        mae_linear_train.append(mean_absolute_error(train_target, linear_clf.predict(train_data)))
        mae_linear_test.append(mean_absolute_error(test_target, linear_clf.predict(test_data)))
    print("Linear mae train = ", np.mean(mae_linear_train))
    print("Linear mae test = ", np.mean(mae_linear_test))

    ridge = Ridge()
    count_lam = list(np.arange(26))
    lambdas = list(map(lambda x: 10 ** (-3 + 0.2 * x), count_lam))
    mae_test = []
    mae_train = []
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.5)
    for elem in lambdas:
        ridge.alpha = elem
        ridge.fit(train_data, train_target)
        mae_train.append(mean_absolute_error(train_target, ridge.predict(train_data)))
        mae_test.append(mean_absolute_error(test_target, ridge.predict(test_data)))
    plt.plot(count_lam, mae_test, label='Test')
    plt.plot(count_lam, mae_train, label='Train')
    plt.xlabel("Lambda")
    plt.legend()
    plt.ylabel("Mean Absolute Error")
    plt.title("Ridge Regression")
    plt.savefig("img/ridge_task4.png")
    plt.close()


if __name__ == '__main__':
    main()
