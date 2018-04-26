import pickle
import numpy as np
from matplotlib import pyplot as plt

with open('delay.txt', 'rb') as fp:
    data = pickle.load(fp)

# fixed bin size
bins = np.arange(-100, 100, 1)  # fixed bin size

plt.xlim([min(data) - 5, max(data) + 5])

plt.hist(data[:20000], bins=bins, alpha=0.5)
plt.title('Random Gaussian data (fixed bin size)')
plt.xlabel('variable X (bin size = 5)')
plt.ylabel('count')

plt.show()
