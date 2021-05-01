from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

FILENAME = "../datasets/nsw74psid1.csv"


def read_data(filename):
    file = read_csv(filename)
    target = file["re78"].to_numpy()
    file.drop(["re78"], axis=1, inplace=True)
    data = file.to_numpy()
    return data, target


def main():
    data, target = read_data(FILENAME)
    train_data, test_data, train_target, test_target = train_test_split(data, target, train_size=0.8)
    regressions = [
        LinearRegression(),
        DecisionTreeRegressor(),
        SVR(kernel="poly")
    ]

    for reg in regressions:
        reg.fit(train_data, train_target)
        prediction = reg.predict(test_data)
        print("\\item {}\'s Score= {}".format(reg.__class__.__name__, mean_squared_error(test_target, prediction)))


if __name__ == '__main__':
    main()