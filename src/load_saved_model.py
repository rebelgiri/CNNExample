from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import model_from_json
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import numpy as np
import pickle
from PIL import Image

from tensorflow import keras

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])

# results = np.argmax(loaded_model.predict_classes(X_test), axis=-1)

print(np.argmax(loaded_model.predict(X_test[100].reshape(1, 28, 28, 1))))
# plt.imshow(X_test[100])
# plt.show()

image = Image.open('../Images/8.jpg').convert('L')
image = np.asarray(image)
# plt.imshow(image)
# plt.show()
image_resized = np.resize(image, (28, 28))
print(image_resized.shape)

plt.imshow(image_resized)
plt.show()

print(np.argmax(loaded_model.predict(image_resized.reshape(1, 28, 28, 1))))


print('end')
