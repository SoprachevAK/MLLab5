from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

FILENAME = "../datasets/cygage.txt"


def read_data():
    file = read_csv('../datasets/cygage.txt', delimiter='\t').to_numpy()
    target = file[:, 0]
    target = target.astype('int')
    data = np.delete(file, [0], axis=1)
    return data, target


def main():
    data, target = read_data()
    plt.plot(data, target, color="red")
    plt.title("Dataset")
    plt.savefig("img/task3.png")
    plt.close()

    clf = LinearRegression()
    accuracy = []
    for i in range(10000):
        train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.40)
        clf.fit(train_data, train_target)
        accuracy.append(clf.score(test_data, test_target))
    print("Score = ", np.mean(accuracy))


if __name__ == '__main__':
    main()
