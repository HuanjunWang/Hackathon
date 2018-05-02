from sklearn import linear_model
from sklearn import svm
from ENV import ENV
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import (TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.preprocessing import PolynomialFeatures

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


def test_ml_model(env, reg, number=100000):
    name = reg.__class__.__name__
    reg.fit(env.X, env.Y)
    state = env.reset()
    total_reward = 0
    for i in range(number):
        adv = reg.predict([state])
        state, reward, end = env.step(advance=adv)
        total_reward += reward
        if end:
            break

    print("[%-24s] Messages:%d number:%d Total Reward:%d" % (name, number, env.STATE_LEN, total_reward))


if __name__ == "__main__":

    kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
             + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

    models = [
        TheilSenRegressor(),
        RANSACRegressor(),
        HuberRegressor(),
        linear_model.LinearRegression(),
        tree.DecisionTreeRegressor(),
        GaussianNB(),
    ]

    steps = [30, 10]
    for step in steps:
        env = ENV(step=step)
        for model in models:
            test_ml_model(env, model)
