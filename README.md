# README.md для Duplicates of PascalVOC2007

##### Запуск autoencoder.py: 

/nn/bin/python3 autoencoder.py(я запускал в google colab)

##### При запуске autoencoder.py:

##### Обучаю свёрточный автоэнкодер на датасете PascalVOC2007(можно брать любой датасет), извлекаю для изображений латентное представление, сохраняю только encoder:

autoencoder = tf.keras.models.Model(inputs=inp, outputs=reconstruction)

autoencoder.compile(optimizer="adamax", loss='mse')

autoencoder.fit(x=train_data, y=train_data, epochs=10, verbose=1)

encoder.save('mencoder.h5')

images = train_data

codes = encoder.predict(images)

assert len(codes) == len(images)

##### В латентном пространстве изображений обучаю BucketedRandomProjectionLSH, сохраняю обученную модель BucketedRandomProjectionLSH:

brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=2.0, numHashTables=3)

model = brp.fit(df)

model.write().overwrite().save('BRP.model')

##### Автоэнкодер:

Автоэнкодер строится при помощи соединения сверточных слоев и слоёв пуллинга, которые уменьшают размерность изображения и извлекают наиболее важные признаки. 

На выходе возвращаются encoder и decoder. 

Для задачи кодирования изображения в вектор, нам нужен слой после автоэнкодера, т.е. векторное представление изображения, которое в дальнейшем будет использоваться для поиска похожих изображений.

##### Применение функции summary() к модели покажет описание работы модели слой за слоем:

Model: "sequential_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_8 (Conv2D)           (None, 224, 224, 32)      896       
                                                                 
 max_pooling2d_8 (MaxPooling2D)  (None, 112, 112, 32)      0                                                                    
                                                                 
 conv2d_9 (Conv2D)           (None, 112, 112, 64)      18496     
                                                                 
 max_pooling2d_9 (MaxPooling2D)  (None, 56, 56, 64)        0                                                                    
                                                                 
 conv2d_10 (Conv2D)          (None, 56, 56, 128)       73856     
                                                                 
 max_pooling2d_10 (MaxPooling2D)  (None, 28, 28, 128)       0                                                                
                                                                 
 conv2d_11 (Conv2D)          (None, 28, 28, 256)       295168    
                                                                 
 max_pooling2d_11 (MaxPooling2D)  (None, 14, 14, 256)       0                                                                 
                                                                 
 flatten_2 (Flatten)         (None, 50176)             0         
                                                                 
 dense_4 (Dense)             (None, 32)                1605664   
                                                                 
=================================================================
Total params: 1994080 (7.61 MB)
Trainable params: 1994080 (7.61 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
Model: "sequential_5"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_5 (Dense)             (None, 50176)             1655808   
                                                                 
 reshape_2 (Reshape)         (None, 14, 14, 256)       0         
                                                                 
 conv2d_transpose_8 (Conv2DTranspose)  (None, 28, 28, 128)       295040                                                          
                                                                 
 conv2d_transpose_9 (Conv2DTranspose)  (None, 56, 56, 64)        73792                                                  
                                                                 
 conv2d_transpose_10 (Conv2DTranspose)  (None, 112, 112, 32)      18464                                                    
                                                                 
 conv2d_transpose_11 (Conv2DTranspose) (None, 224, 224, 3)       867                                                         
                                                                 
=================================================================
Total params: 2043971 (7.80 MB)
Trainable params: 2043971 (7.80 MB)
Non-trainable params: 0 (0.00 Byte)

Автоэнкодер проходит 10 эпох.
_________________________________________________________________
Epoch 1/10

32/32 [==============================] - 208s 6s/step - loss: 0.1154

Epoch 2/10

32/32 [==============================] - 204s 6s/step - loss: 0.0526

Epoch 3/10
32/32 [==============================] - 213s 7s/step - loss: 0.0459

Epoch 4/10

32/32 [==============================] - 204s 6s/step - loss: 0.0423

Epoch 5/10

32/32 [==============================] - 204s 6s/step - loss: 0.0388

Epoch 6/10

32/32 [==============================] - 211s 7s/step - loss: 0.0362

Epoch 7/10

32/32 [==============================] - 206s 6s/step - loss: 0.0350

Epoch 8/10

32/32 [==============================] - 204s 6s/step - loss: 0.0337

Epoch 9/10

32/32 [==============================] - 209s 7s/step - loss: 0.0326

Epoch 10/10

32/32 [==============================] - 205s 6s/step - loss: 0.0317

##### DataFrame с векторными представлениями изображений:

+-----+--------------------+

|index|            features|

+-----+--------------------+

|    0|[4.60763454437255...|

|    1|[-1.4448490142822...|

|    2|[1.00073361396789...|

|    3|[-1.3397672176361...|

|    4|[-1.4091060161590...|

|    5|[7.25881958007812...|

|    6|[5.76556825637817...|

|    7|[-3.4385781288146...|

|    8|[-0.8693913817405...|

|    9|[1.37753295898437...|

|   10|[7.05252933502197...|

|   11|[7.85184335708618...|

|   12|[2.01394200325012...|

|   13|[-2.9907338619232...|

|   14|[-9.3146734237670...|

|   15|[3.81947255134582...|

|   16|[2.68432593345642...|

|   17|[-6.0243558883666...|

|   18|[-4.9376883506774...|

|   19|[-5.5867042541503...|

+-----+--------------------+

only showing top 20 rows

##### Запуск mapper-reducer: 

/hadoop-3.3.6/bin/hadoop jar  /hadoop-3.3.6/share/hadoop/tools/lib/hadoop-streaming-3.3.6.jar -input /PascalVOC2007/Annotations -output /ZuevKP/outputlab -mapper ~/ZuevKP/lab1/mapper.py -reducer ~/ZuevKP/lab1/reducer.py

##### Передаю в mapper обученные модели encoder и BucketedRandomProjectionLSH:

model = load_model("mencoder.h5")

loaded_model = BucketedRandomProjectionLSHModel.load("BRP.model")

##### Применяю approxNearestNeighbors для поиска похожих изображений:

code = model.predict(image_array)

spark = SparkSession.builder.appName("ReadParquet").getOrCreate()

df = spark.read.parquet("data.parquet")

code = np.array(code)

key = Vectors.dense(code)

result = loaded_model.approxNearestNeighbors(df, key, 5)

##### Формат результата:

numbers of Nearest Neighbors: [965, 93, 427, 34, 900] image number: 000059.jpg

номер 965, 93, 427, 34, 900 соответствуют изображениям 000965.jpg 000093.jpg 000427.jpg 000034.jpg 000900.jpg

##### Посмотреть результат:

hdfs dfs -cat /ZuevKP/outputlab/part-00000
