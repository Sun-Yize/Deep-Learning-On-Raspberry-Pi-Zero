import tensorflow as tf
from tensorflow.keras import layers, models
import datetime
import os


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

BATCH_SIZE = 50
ds6 = tf.data.Dataset.list_files("/home/group7/dataset/fruit_flower/train/*/*")
for file in ds6.take(10):
    print(file)

# labels = tf.constant(["apple", " blueberry", "grape", "mango", " pear", " plum", "watermelon", "banana", "cherry", " lemon", "orange", "pineapple", "strawberry"])


def load_image(img_path,size = (224,224)):
    label = tf.cast(tf.compat.v1.string_to_number(tf.strings.split(img_path, sep='/',)[6]), tf.int8)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img,size)/255.0
    return(img,label)


ds_train = tf.data.Dataset.list_files("/home/group7/dataset/fruit_flower/train/*/*") \
           .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
           .shuffle(buffer_size = 1000).batch(BATCH_SIZE) \
           .prefetch(tf.data.experimental.AUTOTUNE)

ds_test = tf.data.Dataset.list_files("/home/group7/dataset/fruit_flower/test/*/*") \
           .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
           .batch(BATCH_SIZE) \
           .prefetch(tf.data.experimental.AUTOTUNE)

inputs = layers.Input(shape=(224,224,3))
x = layers.Conv2D(32,kernel_size=(3,3),padding="same")(inputs)
x = layers.Activation('relu')(x)

x = layers.Conv2D(32,kernel_size=(3,3),padding="same")(x)
x = layers.Activation('relu')(x)
x = layers.MaxPool2D(pool_size=2,strides=2)(x)
x = layers.Conv2D(64,kernel_size=(3,3),padding="same")(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(64,kernel_size=(3,3),padding= "same")(x)
x = layers.Activation('relu')(x)
x = layers.MaxPool2D(pool_size=2,strides=2)(x)
### 继续构建卷积层和池化层，这次核数量设置成128
x = layers.Conv2D(128,kernel_size=(3,3),padding="same")(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(128,kernel_size=(3,3),padding="same")(x)
x = layers.Activation('relu')(x)
x = layers.MaxPool2D()(x)
x = layers.Flatten()(x)##数据扁平化处理
x = layers.Dense(128,activation='relu')(x)
x = layers.Dense(64,activation='relu')(x)
x = layers.Dropout(rate=0.3)(x)
outputs = layers.Dense(30,activation = 'softmax')(x)

model = models.Model(inputs = inputs,outputs = outputs)

model.summary()


stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join('data', 'autograph', stamp)
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
model.compile(
         optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
         loss=tf.keras.losses.sparse_categorical_crossentropy,
         metrics=["accuracy"]
    		 )
history = model.fit(ds_train,epochs= 500,validation_data=ds_test,
                     callbacks = [tensorboard_callback],workers = 4)

# # 保存权重，该方式仅仅保存权重张量
model.save_weights('./data/tf_model_weights.ckpt',save_format = "tf")
#
# # 保存模型结构与模型参数到文件,该方式保存的模型具有跨平台性便于部署
#
model.save('./data/tf_model_savedmodel', save_format="tf")
print('export saved model.')

model_loaded = tf.keras.models.load_model('./data/tf_model_savedmodel')
model_loaded.evaluate(ds_test)
