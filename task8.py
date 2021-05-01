import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt

FILENAME = "../datasets/svmdata6.txt"


def main():
    file = read_csv(FILENAME, delimiter='\t').to_numpy()
    data = file[:, 0].reshape(-1, 1)
    target = file[:, 1]

    clf_svr = SVR(kernel="rbf", C=1)
    train_data, test_data, train_target, test_target = train_test_split(data, target, train_size=0.8)

    epsilon = np.array(np.linspace(0, 2, num=20))
    MSE = []
    for elem in epsilon:
        clf_svr.epsilon = elem
        clf_svr.fit(train_data, train_target)
        prediction = clf_svr.predict(test_data)
        MSE.append(mean_squared_error(test_target, prediction))
    plt.plot(epsilon, MSE)
    plt.xlabel("Epsilon")
    plt.ylabel("Mean Square Error")
    plt.savefig("img/mse_task8.png")
    plt.close()


if __name__ == '__main__':
    main()