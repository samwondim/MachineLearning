import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

data = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = Sequential([tf.keras.layers.Flatten(input_shapt=(28,28)),
                   Dense(128, activation=tf.nn.relu),
                   Dense(10, activation=tf.nn.softmax)
                    ])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)

