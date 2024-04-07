#!/nn/bin/python3

import sys
import os
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing import image
from pyarrow import fs
from PIL import Image
import io

cifar = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

fs = fs.HadoopFileSystem("hdfs://localhost:9000?user=guest")

model = load_model('cifar10_model.h5')

k = 0
n = 0
for line in sys.stdin:
    pred, number = line.split('\t')
    number = number.strip()
    f = fs.open_input_file("/PascalVOC2007/JPEGImages/"+number)
    tmp = f.read()
    img_array = bytearray(tmp)
    image = Image.open(io.BytesIO(img_array))
    img = np.array(image)/255.0
    img = keras.preprocessing.image.smart_resize(img,(32,32))
    img = np.array([img.astype("float32")])

    sys.stdout = open(os.devnull, 'w')
    predict = model.predict(img, verbose=0)
    sys.stdout = sys.__stdout__
    predicted = np.argmax(predict)
    n += 1
    if pred == cifar[predicted]:
        k += 1
    print(pred, cifar[predicted], sep='\t')
print("acc", n/k)
