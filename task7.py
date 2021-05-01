from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

FILENAME = "../datasets/cars.csv"


def main():
    file = read_csv(FILENAME).to_numpy()
    target = file[:, 1].astype("int")
    data = file[:, 0].reshape(-1, 1).astype("int")

    plt.scatter(data, target, marker="o", color="r", label='Dataset\'s points')
    plt.xlabel("Speed")
    plt.ylabel("Distance")

    clf = LinearRegression()
    clf.fit(data, target)
    prediction = clf.predict(data)
    plt.plot(data, prediction, color='green', label='Linear Regression\'s points')

    neig_clf = KNeighborsRegressor()
    neig_clf.fit(data, target)
    prediction = neig_clf.predict(data)
    plt.plot(data, prediction, color='blue', label='KNeighbors Regression\' points')

    plt.legend()
    plt.savefig("img/dataset_task7.png")
    plt.close()

    prediction_40 = neig_clf.predict([[40]])
    print(prediction_40[0])


if __name__ == '__main__':
    main()