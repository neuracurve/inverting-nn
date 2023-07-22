
import tensorflow as tf
from tensorflow.python.ops import math_ops
import keras.backend as K
import numpy as np

print(f'Tensorflow version: {tf.__version__}')

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import math

# Hyperparameters
batch_size = 1
epochs = 4000
optimizer = Adam(learning_rate=0.01, decay=0.0005)
# optimizer = Adam(learning_rate=0.0001)
weight_init = RandomNormal()

# Build model
inputs = tf.keras.Input(shape=(1,))
x = layers.Dense(512, activation='gelu', name='H1', kernel_initializer=weight_init,\
                 kernel_regularizer=None)(inputs)
x = layers.Dense(512, activation='gelu', name='H2', kernel_initializer=weight_init,
                 kernel_regularizer=None)(x)
output = layers.Dense(1, activation='linear', name='Out', kernel_initializer=weight_init)(x)
model = tf.keras.Model(inputs, output)

# function to be inverted
def f_tbi(x):
    return x**2

x_domain = True  # True for positive x domain
if x_domain:
    sign = 1
else:
    sign = -1

y_coloc = np.arange(0.0, 2.0, 0.01)  # define domain of inverse function
# y_coloc = np.arange(0.0, 0.2, 0.005)  # define domain [near zero]
# x_coloc = f_tbi(y_coloc)  # for plot of f(x)

# initial condition (point on inverse function)
# y_init = np.array([1.0])
# x_init = np.array([1.0])
y_init = np.array([1.0])
x_init = np.array([1.0]) * sign

# Step function
def step(y, y_init, x_init):
    y = tf.convert_to_tensor(y)
    y = tf.reshape(y, [batch_size, 1])  # required by keras input
    y = tf.Variable(y)  # , name='x_co'
    with tf.GradientTape(persistent=False) as tape:  #false - no higher gradients

        #model_loss1: initial condition y_init @ x_init -> f(x) initial condition
        pred_init = model(y_init)
        model_loss1 = math_ops.squared_difference(pred_init, x_init)

        # model_loss2: collocation points
        pred_x = model(y)
        # dfdx = tape.gradient(pred_y, x_co)  # f(x)'
        func = f_tbi(pred_x)
        func = tf.cast(func, tf.float64)
        residual = func - y
        model_loss2 = K.mean(math_ops.square(residual), axis=-1)
        model_loss2 = tf.cast(model_loss2, tf.float32)

        #total loss
        model_loss = model_loss1 + model_loss2

        trainable = model.trainable_variables
        model_gradients = tape.gradient(model_loss, trainable)

        # Update model
        optimizer.apply_gradients(zip(model_gradients, trainable))
        return np.mean(model_loss)

# Training loop
bat_per_epoch = math.floor(len(y_coloc) / batch_size)
loss = np.zeros(epochs)
for epoch in range(epochs):
    print(f'epoch: {epoch}  model(0): {model(np.array([0.0]))}')
    for i in range(bat_per_epoch):
        n = i * batch_size
        loss[epoch] = step(y_coloc[n:n + batch_size], y_init, x_init)

# compare PINN results vs analytical results
n_pts = 100  # number test points
y_test = np.zeros(n_pts)
x_test = np.zeros(n_pts)
x_calc = np.zeros(n_pts)
y_calc = np.zeros(n_pts)

for i in range(n_pts):
    y_test[i] = i / 50
    x_test[i] = model.predict([y_test[i]])
    y_calc[i] = f_tbi(y_test[i])  # f(x)
    x_calc[i] = np.sqrt(y_test[i])  # inverse f(x)

plt.plot(y_test, x_test, 'r', y_test[5:100:10], x_calc[5:100:10] * sign, 'sb', y_test, y_calc, 'g')  # 'sg'
# plt.plot(x_test, y_test, 'r')
plt.ylim([-1.5, 4.0])
plt.xlim([-0.1, 2.0])
# plt.hlines(2.0, 0.0, 5.0, color='m', linestyle='dotted')
plt.text(1.4, 2.5, '$f(x)$', va='center')
plt.text(1.5, 0.9*sign, '$f^{-1}(y)$', va='center')
plt.title('Function and Inverse Function')
plt.xlabel('green: x     red: y')
plt.ylabel('green: $f(x)$     red: $f^{-1}(y)$')
plt.show()

plt.plot(loss)
plt.yscale('log')
plt.show()
