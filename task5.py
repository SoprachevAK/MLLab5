import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

FILENAME = "../datasets/eustock.csv"


def read_data(filename):
    file = read_csv(filename)
    DAX_lst = file["DAX"].to_numpy()
    SMI_lst = file["SMI"].to_numpy()
    CAC_lst = file["CAC"].to_numpy()
    FTSE_lst = file["FTSE"].to_numpy()
    return DAX_lst, SMI_lst, CAC_lst, FTSE_lst


def draw_data(lst):
    titles = ['DAX', 'SMI', 'CAC', 'FTSE']
    xticks = [i for i in range(1, len(lst[0]) + 1)]
    reals = []
    plt.figure(figsize=(20, 10))
    for column in lst:
        real, = plt.plot(xticks, column)
        reals.append(real)
    plt.legend(reals, titles)
    plt.grid(True)
    plt.xticks(())
    plt.savefig("img/task5_dataset.png")
    plt.close()


def draw_regression(lst):
    titles = ['DAX', 'SMI', 'CAC', 'FTSE']
    xticks = np.array([i for i in range(1, len(lst[0]) + 1)])
    plt.figure(figsize=(20, 10))
    for column, title in zip(lst, titles):
        clf = LinearRegression()
        column = column.reshape(-1)
        xticks_reshaped = xticks.reshape(-1, 1)
        clf.fit(xticks_reshaped, column)
        pred = clf.predict(xticks_reshaped)
        plt.plot(xticks_reshaped, pred)
    plt.legend(titles)
    plt.grid(True)
    plt.xticks(())
    plt.savefig("img/task5_regression.png")
    plt.close()


def main():
    DAX_lst, SMI_lst, CAC_lst, FTSE_lst = read_data(FILENAME)
    lst = np.array([DAX_lst, SMI_lst, CAC_lst, FTSE_lst])
    draw_data(lst)
    draw_regression(lst)


if __name__ == '__main__':
    main()
