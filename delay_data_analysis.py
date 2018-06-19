import pickle
import numpy as np
from matplotlib import pyplot as plt

with open('delay.txt', 'rb') as fp:
    data = pickle.load(fp)

plt.bar(data[:100000])


# fixed bin size
i = 0
while i < 500000:
    plt.plot(data[i:i+500], 'b')
    i += 1000
    plt.show()
