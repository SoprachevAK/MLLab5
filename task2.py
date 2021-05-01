from pandas import read_csv
import numpy as np
from itertools import chain, combinations
from sklearn.linear_model import LinearRegression

FILENAME = "../datasets/reglab.txt"


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def read_data(filename):
    file = read_csv(filename, delimiter='\t').to_numpy()
    target = file[:, 0]
    data = np.delete(file, [0], axis=1)
    return data, target


def main():
    data, target = read_data(FILENAME)
    n = list(np.arange(len(data[0])))
    sets = list(powerset(n))[1:]
    sets = [list(elem) for elem in sets]
    clf = LinearRegression()
    res = []
    labels = np.array(('X_1', 'X_2', 'X_3', 'X_4'))
    for cols in sets:
        cur_data = data[:, cols]
        clf.fit(cur_data, target)
        predict = clf.predict(cur_data)
        res.append((cols, np.sum(np.square(target - predict))))

    res = sorted(res, key=lambda res: res[1])
    for i, r in zip(range(len(res)), res):
        arr = labels[r[0]]
        str = ""
        for elem in arr:
            str += elem + ", "
        print('\\item ${}$ = {}'.format(str, r[1]))


if __name__ == '__main__':
    main()