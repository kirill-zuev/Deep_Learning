#!/nn/bin/python3

import sys
import os
import numpy as np
import keras
import numpy as np 
from keras import Input 
from keras.datasets import mnist 
from keras.models import Sequential 
from keras.utils import to_categorical
from keras.layers import Dense, Activation
from keras.metrics import CategoricalAccuracy
from keras.preprocessing import image
from keras.models import load_model

model = load_model('mnist_model.h5')

(_, _), (x_test, y_test) = mnist.load_data()

x_test = np.reshape(x_test, (-1, 28*28)).astype("float32") / 255.0
y_test = to_categorical(y_test)
line=sys.stdin
model = load_model('mnist_model.h5')

for i in range(x_test.shape[0]):
    sys.stdout = open(os.devnull, 'w')
    prediction = model.predict(np.array([x_test[i]]))
    sys.stdout = sys.__stdout__
    predicted = np.argmax(prediction)
    print(predicted, np.argmax(y_test[i]), sep='\t')
