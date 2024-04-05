#!/nn/bin/python3

import sys
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.datasets import mnist
import sys

for line in sys.stdin:
    pred, true = line.split('\t')
    print("pred:", pred, "true:", true)

