#!/nn/bin/python3

import sys
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

model = load_model('mnist_model.h5')

for line in sys.stdin:
    image_path = line.strip()
    img = image.load_img(image_path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    img_array = img_array.reshape(1, 784)
    sys.stdout = open(os.devnull, 'w')
    prediction = model.predict(img_array)
    sys.stdout = sys.__stdout__
    predicted = np.argmax(prediction)
    print(predicted, sep='\t')
