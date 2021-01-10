import tensorflow as tf
from tensorflow.keras import layers, models
import datetime
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
BATCH_SIZE = 128


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

Mo = tf.keras.applications.MobileNetV2(
    	input_shape=(224,224,3),
	include_top=False,
	weights='imagenet')

model = models.Sequential()
model.add(Mo)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(rate=0.3))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(rate=0.3))
model.add(layers.Dense(30, activation='sigmoid'))
model.summary()

for layer in Mo.layers[:20]:
   layer.trainable=False
for layer in Mo.layers[20:]:
   layer.trainable=True

stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join('model', 'autograph', stamp)
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

checkpoint_dir = os.path.join('model', 'MoblieNetV2')
checkpoint_path = os.path.join('model', 'MoblieNetV2', stamp)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=20)


latest = tf.train.latest_checkpoint(checkpoint_dir)
#model.load_weights(latest)
model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=["accuracy"]
    )

history = model.fit(ds_train,epochs= 1,validation_data=ds_test,
                    callbacks = [tensorboard_callback, cp_callback])

# 保存模型结构与模型参数到文件,该方式保存的模型具有跨平台性便于部署

model.save('./data/tf_model_savedmodel', save_format="tf")
print('export saved model.')

model_loaded = tf.keras.models.load_model('./data/tf_model_savedmodel')
model_loaded.evaluate(ds_test)

