import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

FILENAME = "../datasets/JohnsonJohnson.csv"


def read_data(filename):
    file = read_csv(filename, delimiter=',').to_numpy()
    quarters = [[] for i in range(4)]
    for elem in file:
        quarters[int(elem[0][-1]) - 1].append(elem[1])
    quarters = np.array([np.array(elem).reshape(-1, 1) for elem in quarters])
    return quarters


def main():
    quarters = read_data(FILENAME)
    x_axis = range(len(quarters[0]))
    years = np.arange(1960, 1981)
    plt.figure(figsize=(20, 10))
    for q in quarters:
        plt.plot(x_axis, q)
    plt.xticks(x_axis, years)
    plt.legend(('Q1', 'Q2', 'Q3', 'Q4'))
    plt.grid(True)
    plt.savefig("img/dataset_task6.png")
    plt.close()

    all_years = np.sum(np.concatenate(quarters, axis=1), axis=1).reshape(-1, 1)
    plt.figure(figsize=(20, 10))
    prediction_2016 = []
    clf = LinearRegression()
    yreshaped = years.reshape(-1, 1)
    for q in quarters:
        clf.fit(yreshaped, q.reshape(-1))
        pred = clf.predict(yreshaped)
        plt.plot(years, pred)
        prediction_2016.append(clf.predict([[2016]])[0])

    plt.xticks(years, [str(i) for i in years])
    plt.legend(('Q1', 'Q2', 'Q3', 'Q4'))
    plt.grid(True)
    plt.savefig("img/linear_task6.png")
    plt.close()
    for p, i in zip(prediction_2016, range(1, 5)):
        print('\\item Q' + str(i) + " ", p)
    clf.fit(yreshaped, all_years)
    print(clf.predict([[2016]])[0])


if __name__ == '__main__':
    main()