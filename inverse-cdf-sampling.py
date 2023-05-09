
import tensorflow as tf
from tensorflow.python.ops import math_ops
import keras.backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

# load inverse cdf neural network model
dir_path = '[insert path here]'
inverse_cdf_model = load_model(dir_path + "inverse-cdf-model.h5", compile=False)


# function to sample from inverse_cdf_model neural net
def get_sample(probability):  # probability -> numpy array
    sample = inverse_cdf_model(probability)
    return sample

# plot inverse cdf + pts from scipy.stats.ppf
# n_pts = 100  # number test points
y_test = np.arange(0, 1, 0.01)
x_test = get_sample(y_test)
y_test2 = np.arange(0.02, 0.99, 0.024)
x_calc = norm.ppf(y_test2)

plt.plot(y_test, x_test, 'r', y_test2, x_calc, '.g')
plt.title('red: quantile (neural net)      green: quantile (SciPy)')
plt.xlabel('probability')
plt.ylabel('x')
plt.show()

# inverse transform sampling
# get uniform distribution samples
s_uniform = np.random.default_rng().uniform(0, 1, 30000)

# sample from inverse CDF neural network with uniform samples
t = get_sample(s_uniform)
t = np.reshape(t, -1)

# plot pdf histogram of inverse CDF samples
plt.hist(t, 500, histtype='step', density=True)

# create pdf of normal distribution for comparison
mu = 0.0
sigma = 1.0
def normal_pdf(x):
    return (1 / (sigma * np.sqrt(2 * math.pi)))\
           * np.exp(-0.5 * ((x - mu ) / sigma)**2)


x_test = np.arange(-2.5, 2.5, 0.05)
y_normal = normal_pdf(x_test)
plt.plot(x_test, y_normal, color='r')

plt.xlim([-2.3, 2.3])
plt.ylabel('probability density (pdf)')
plt.xlabel('x')
plt.title('Inverse CDF Samples')
plt.show()

