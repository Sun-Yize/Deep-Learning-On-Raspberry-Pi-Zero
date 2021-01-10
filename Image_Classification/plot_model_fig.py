import tensorflow as tf
from tensorflow.keras import layers, models
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

BATCH_SIZE = 128
EPOCHS = 20
datapath = '/home/group7/tensorflow_learn/train/'

def load_image(img_path,size = (224,224)):
    label = tf.cast(tf.compat.v1.string_to_number(tf.strings.split(img_path, sep='/',)[6]), tf.int8)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img,size)/255.0
    return(img,label)


ds_train = tf.data.Dataset.list_files("/home/group7/dataset/fruit_flower/train/*/*.jpg") \
           .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
           .shuffle(buffer_size = 1000).batch(BATCH_SIZE) \
           .prefetch(tf.data.experimental.AUTOTUNE)

ds_test = tf.data.Dataset.list_files("/home/group7/dataset/fruit_flower/test/*/*.jpg") \
           .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
           .batch(BATCH_SIZE) \
           .prefetch(tf.data.experimental.AUTOTUNE)

# tf.keras.applications导入模型
Mo = tf.keras.applications.MobileNetV2(
    	input_shape=(224,224,3),
		include_top=False)
Mo.trainable=True

model = models.Sequential()
model.add(Mo)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(30, activation='sigmoid'))
model.summary()

import datetime


stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
callback_dir = datapath+'model/callback/'+stamp
tensorboard_callback = tf.keras.callbacks.TensorBoard(callback_dir, histogram_freq=1)

checkpoint_path = datapath+'model/checkpoint/'+stamp
model_save = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, 
                verbose=1, 
                save_weights_only=True,
                period=20)

early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.001, 
                           patience = 5, mode = 'min', verbose = 1,
                           restore_best_weights = True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, 
                              patience = 2, min_delta = 0.001, 
                              mode = 'min', verbose = 1)

model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.0005),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=["accuracy"])

history = model.fit(ds_train,epochs=EPOCHS,validation_data=ds_test,
                    callbacks = [tensorboard_callback, model_save, early_stop, reduce_lr])


from matplotlib import pyplot as plt

plt.figure(1)
train_metrics = history.history["loss"]
val_metrics = history.history['val_loss']
epochs = range(1, len(train_metrics) + 1)
plt.plot(epochs, train_metrics, 'bo--')
plt.plot(epochs, val_metrics, 'ro-')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend(["train_loss", 'val_loss'])
plt.savefig('./test1.jpg')

plt.figure(2)
train_metrics = history.history["accuracy"]
val_metrics = history.history['val_accuracy']
epochs = range(1, len(train_metrics) + 1)
plt.plot(epochs, train_metrics, 'bo--')
plt.plot(epochs, val_metrics, 'ro-')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.legend(["train_accuracy", 'val_accuracy'])
plt.savefig('./test2.jpg')
