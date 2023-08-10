import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# define the neuron 
l0 = Dense(units=1, input_shape=[1])

# pass the neuron to the model.
model = Sequential([l0])
model.compile(optimizer="sgd", loss="mean_squared_error")

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype="float")
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype="float")

model.fit(xs, ys, epochs=500)

print(model.predict([10]))

print("Here are the values I've learned: {}".format(l0.get_weights()))
