import numpy as np
from scipy import stats

x = np.linspace(1,10,10)
y = 2*x + np.random.random()

print(stats.pearsonr(x,y)[0])