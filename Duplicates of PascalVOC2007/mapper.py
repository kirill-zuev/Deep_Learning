#!/nn/bin/python3

import sys
import numpy as np
import keras
from keras.models import load_model
from pyarrow import fs
from PIL import Image
import io
import cv2
from pyspark.ml.feature import BucketedRandomProjectionLSHModel
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

def get_values(s, ws, i1):
    ans = []
    for w in ws:
        op = ''.join(['<',w,'>'])
        cl = ''.join(['</',w,'>'])
        i0 = s.find(op,i1)
        i1 = s.find(cl,i0)
        sub = s[i0+len(op):i1]
        ans.append(sub)
    return ans, i1

def parse(s):
    fname, i1 = get_values(s,["filename"],0)
    fname = fname[0]
    return fname

lines = []
for line in sys.stdin:
    lines.append(line)

s = ''.join(lines)

number = parse(s)

fs = fs.HadoopFileSystem("hdfs://localhost:9000?user=guest")

model = load_model("mencoder.h5")

number = number.strip()
f = fs.open_input_file("/PascalVOC2007/JPEGImages/"+number)
tmp = f.read()
img_array = bytearray(tmp)
image = Image.open(io.BytesIO(img_array))
image_array = np.array(image)
image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
image_array = cv2.resize(image_array, (224,224))
image_array = image_array.reshape(224, 224, 3)
image_array = image_array.astype('float32')
image_array /= 255
image_array = np.array([image_array])

code = model.predict(image_array)

spark = SparkSession.builder.appName("ReadParquet").getOrCreate()

df = spark.read.parquet("data.parquet")

loaded_model = BucketedRandomProjectionLSHModel.load("BRP.model")

code = np.array(code)
key = Vectors.dense(code)
result = loaded_model.approxNearestNeighbors(df, key, 3)

neighbor_indices = result.select("index").rdd.flatMap(lambda x: x).collect()

print(neighbor_indices, number)
