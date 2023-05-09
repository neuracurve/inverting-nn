
import tensorflow as tf
from tensorflow.python.ops import math_ops
import keras.backend as K
import numpy as np
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import math

# Hyperparameters
batch_size = 1
epochs = 500
optimizer = Adam(learning_rate=0.001)
weight_init = RandomNormal()

# Build model
inputs = tf.keras.Input(shape=(1,))
x = layers.Dense(256, activation='gelu', name='H1', kernel_initializer=weight_init,\
                 kernel_regularizer=None)(inputs)
x = layers.Dense(256, activation='gelu', name='H2', kernel_initializer=weight_init,
                 kernel_regularizer=None)(x)
output = layers.Dense(1, activation='linear', name='Out', kernel_initializer=weight_init)(x)
model = tf.keras.Model(inputs, output)

# load function model
dir_path = '[insert path here]'
cdf_model = load_model(dir_path + "normal-cdf-model.h5", compile=False)
test_var = np.array([0])
print(f'cdf_model test (approx. 0.5): {cdf_model.predict(test_var)}')

# function to be inverted
def f_tbi(x):
    prob = cdf_model(x)
    return prob


y_coloc = np.arange(0.1, 0.9, 0.01)  # define domain of inverse function
y_coloc_densel = np.arange(0.0, 0.1, 0.001)
y_coloc_denser = np.arange(0.9, 1.0, 0.001)
y_coloc = np.append(y_coloc, y_coloc_densel)
y_coloc = np.append(y_coloc, y_coloc_denser)
rng = np.random.default_rng()
rng.shuffle(y_coloc)

# initial condition (point on inverse function)
# (stabilizes training)
x_init = np.array([0.0])
y_init = cdf_model.predict(x_init)

# Step function
def step(y, y_init, x_init):
    y = tf.convert_to_tensor(y)
    y = tf.reshape(y, [batch_size, 1])  # required by keras input
    y = tf.Variable(y)
    with tf.GradientTape(persistent=False) as tape:  #false - no higher gradients

        #model_loss1: initial condition y_init @ x_init -> f(x) initial condition
        pred_init = model(y_init)
        model_loss1 = math_ops.squared_difference(pred_init, x_init)

        # model_loss2: collocation points
        pred_x = model(y)
        func = f_tbi(pred_x)
        func = tf.cast(func, tf.float64)
        residual = func - y
        model_loss2 = K.mean(math_ops.square(residual), axis=-1)
        model_loss2 = tf.cast(model_loss2, tf.float32)

        #total loss
        model_loss = model_loss1 + model_loss2 * 10

        trainable = model.trainable_variables
        model_gradients = tape.gradient(model_loss, trainable)

        # Update model
        optimizer.apply_gradients(zip(model_gradients, trainable))
        return np.mean(model_loss)

# Training loop
bat_per_epoch = math.floor(len(y_coloc) / batch_size)
loss = np.zeros(epochs)
for epoch in range(epochs):
    print(f'epoch: {epoch}  loss: {loss[epoch-1]}')
    for i in range(bat_per_epoch):
        n = i * batch_size
        loss[epoch] = step(y_coloc[n:n + batch_size], y_init, x_init)

# save model
dir_path = '[insert path here]'
model.save(dir_path + 'inverse-cdf-model.h5')

# plot results
y_test = np.arange(0, 1, .01)
x_test = model.predict([y_test])

plt.plot(y_test, x_test, 'r')
plt.title('Inverse CDF Function')
plt.xlabel('probability')
plt.ylabel('x')
plt.show()

plt.plot(loss)
plt.yscale('log')
plt.show()
