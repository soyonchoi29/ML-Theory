import numpy as np
import math


# returns X = (n, d), y = (n, ) of iid samples for linreg w l2 regularization
def get_favorite_data_1(n):
    # returns points on a line of slope 2.5, y-intercept 1
    m = 2.5
    b = 1
    X = (np.random.rand(n, 1) * 100)-50  # x ranges from -50 to 50
    y = m*X[:, 0]+b
    return X, y


# returns X = (n, d), y = (n, ) of iid samples for kernelridge w deg3 polynomial
def get_favorite_data_2(n):
    X = (np.random.rand(n, 1) * 100)-50  # x ranges from -50 to 50
    y = np.multiply(X[:, 0]+4, np.multiply(X[:, 0]+2, X[:, 0]+3))
    return X, y


# returns X = (n, d), y = (n, ) of iid samples for linreg w deg10 polynomial
def get_favorite_data_3(n):
    X = (np.random.rand(n,) * 100)-50  # x ranges from -50 to 50
    powers = np.vander(X, 11)[:, :-1]
    coeffs = np.random.randint(low=1, high=5, size=10)
    y = coeffs.reshape((1, 10))@powers.T
    y = y.reshape((n,))
    return X.reshape((n, 1)), y


# returns X = (n, d), y = (n, ) of iid samples for linreg w gaussian kernel
def get_favorite_data_4(n):
    # angles = np.random.rand(n, d) * np.pi  # uniformly from 0 to pi
    # cosines = np.concatenate([np.cos(angles), np.ones((n, 1))], axis=1)
    #
    # sines = np.concatenate([np.ones((n, 1)), np.sin(angles)], axis=1)
    # sines = np.log(sines) @ np.triu(np.ones((d+1, d+1)))
    # sines = np.exp(sines)
    # sines[:, -1] = np.multiply(sines[:, -1], (np.random.randint(0, 2, size=(n,)) * 2) - 1)
    #
    # rad = np.ones((n, 1))*r
    # rad = rad.repeat(d+1, axis=0).reshape(cosines.shape)
    #
    # points = np.multiply(np.multiply(cosines, sines), rad)
    # X = points[:, :-1].reshape((n, d))
    # y = np.abs(points[:, -1].reshape((n, )))

    X = (np.random.rand(n,) * 100)-50  # x ranges from -50 to 50
    X = X.reshape((n, 1))
    y = np.exp(-1*(X**2))
    return X, y


def get_favorite_data_5(n):
    pass


def get_favorite_data_6(n):
    pass


def get_favorite_data_7(n):
    pass


def get_favorite_data_8(n):
    pass


def example_get_favorite_data():
    # Two, far apart, spherical Gaussian blobs
    d = 5
    
    mu0 = np.array([-5 for i in range(d)])
    mu1 = np.array([ 5 for i in range(d)])

    y = np.random.binomial(1, 0.5)  # flip a coin for y

    if y == 0:
        x = np.random.multivariate_normal(mean = mu0, cov = np.eye(d))
    else:
        x = np.random.multivariate_normal(mean = mu1, cov = np.eye(d))

    return x, y


def get_lots_of_favorite_data(n=100, data_fun=example_get_favorite_data):
    pts = [data_fun() for _ in range(n)]
    Xs, ys = zip(*pts)
    X = np.array(Xs)
    y = np.array(ys)
    return X, y


if __name__ == "__main__":
    print("Here are some points from example_get_favorite_data:")
    for i in range(4):
        x, y = example_get_favorite_data()
        print(f"\tx: {x}")
        print(f"\ty: {y}")

    print("And here we use get_lots_of_favorite_data to obtain X and y:")
    X, y = get_lots_of_favorite_data(10, example_get_favorite_data)

    print("X:")
    print(X)
    print("y:")
    print(y)
