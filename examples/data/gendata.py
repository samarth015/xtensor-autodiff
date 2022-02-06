from sklearn.datasets import make_regression

data = make_regression(n_samples=500, n_features=5, noise=3)

for X, y in zip(data[0], data[1]):
    print(*X, y, sep=' ', end = '\n')
