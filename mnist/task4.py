import keras
import numpy as np 
from keras import Input 
from keras.datasets import mnist 
from keras.models import Sequential 
from keras.utils import to_categorical
from keras.layers import Dense, Activation
from keras.metrics import CategoricalAccuracy

batch_size = 32
epochs = 3

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.reshape(x_train, (-1, 28*28)).astype("float32") / 255.0
x_test = np.reshape(x_test, (-1, 28*28)).astype("float32") / 255.0
x_train = np.reshape(x_train, (-1, 28*28)).astype("float32") / 255.0
x_test = np.reshape(x_test, (-1, 28*28)).astype("float32") / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential([
    keras.Input(shape=(28*28,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10),
    keras.layers.Activation('softmax'),
])

train_acc_metric = CategoricalAccuracy()
val_acc_metric = CategoricalAccuracy()

from keras.optimizers import Adam
model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=1e-3),
        metrics=[train_acc_metric]
)

model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1
)

model.save('mnist_model.h5')
#image = keras.utils.load_img('/home/guest/ZuevKP/5.png').convert('L')
image = keras.utils.load_img('/nn/5.png').convert('L')
input_arr = [np.reshape(keras.utils.img_to_array(image), (28*28,)).astype("float32") / 255.0]
print(model.predict(np.array(input_arr))) 
