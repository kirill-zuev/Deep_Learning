import os
import cv2

import numpy as np

train_path = './data/train/train_256'
test_path = './data/test/test_256'

train_path0 = './data/train_prepr/'
test_path0 = './data/test_prepr/'

image_paths = []
for root, dirs, files in os.walk(test_path):
    for file in files:
        if file.lower().endswith('.jpg'):
            image_paths.append(os.path.join(root, file))

for path in image_paths:
    path0 = path.split('/')[-1]
    image = cv2.imread(path)
    h, w = image.shape[:2]
    x, y = w // 2, h // 2
    r = 100
    mask = np.zeros_like(image)
    cv2.circle(mask, (x, y), r, (255, 255, 255), thickness=-1)
    result = cv2.bitwise_and(image, mask)
    print(result.shape)
    cv2.imwrite(test_path0+path0, result)