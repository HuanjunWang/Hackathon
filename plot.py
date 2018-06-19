import pickle

from matplotlib import pyplot as plt


with open("./saver/l1_300_l2_300_lr_0.0300_adam_model_1.lose.history", 'rb') as lhf:
    lh = pickle.load(lhf)
    print(lh)


plt.plot(range(len(lh)), lh)
plt.show()