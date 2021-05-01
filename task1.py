from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

FILENAME = "../datasets/reglab1.txt"


def read_data(filename, var):
    file = read_csv(filename, sep="\t")
    target = file[var['t']].to_numpy()
    file.drop([var['t']], axis=1, inplace=True)
    data = file.to_numpy()
    return data, target


def read_data_dic(filename, var):
    file = read_csv(filename, sep="\t")
    target = file[var['t']].to_numpy()
    file.drop([var['r'], var['t']], axis=1, inplace=True)
    data = file.to_numpy()
    return data, target


def main():
    classifier = LinearRegression()
    vars = [{'d': 'x,z', 't': 'y'}, {'d': 'y,z', 't': 'x'}, {'d': 'x,y', 't': 'z'}]
    for var in vars:
        data, target = read_data(FILENAME, var)
        train_data, test_data, train_target, test_target = train_test_split(data, target, train_size=0.8)
        classifier.fit(train_data, train_target)
        print("Func {}({}) Score = {}".format(var['t'], var['d'], classifier.score(test_data, test_target)))

    vars = [{'d': 'y', 't': 'x', 'r': 'z'}, {'d': 'z', 't': 'y', 'r': 'x'}, {'d': 'x', 't': 'z', 'r': 'y'}]
    for var in vars:
        data, target = read_data_dic(FILENAME, var)
        train_data, test_data, train_target, test_target = train_test_split(data, target, train_size=0.8)
        classifier.fit(train_data, train_target)
        print("Func {}({}) Score = {}".format(var['t'], var['d'], classifier.score(test_data, test_target)))


if __name__ == '__main__':
    main()