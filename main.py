"""Enhance Image Using Tensorflow, Fine-Tuning SRCNN model and also using RGB image
    instead of GreyScale image"""
import tensorflow as tf
from keras.layers import Input, Conv2D, UpSampling2D
from keras.models import Model

train_path = 'Train_x'
output_path = 'Train_Y'
valid_path = 'Valid_x'
valid_output_path = 'Valid_Y'

train_img = tf.data.Dataset.list_files(train_path + '/*.*', shuffle=False)
output_img = tf.data.Dataset.list_files(output_path + '/*.*', shuffle=False)
valid_img = tf.data.Dataset.list_files(valid_path + '/*.*', shuffle=False)
valid_output = tf.data.Dataset.list_files(valid_output_path + '/*.*', shuffle=False)

Training_images = tf.data.Dataset.zip((train_img, output_img))
Validating_images = tf.data.Dataset.zip((valid_img, valid_output))


def load_image(image_path, output_image_path):
    """Load the image to create the pipline and process the image by resizing and normalizing
    :param:
        image_path: the path of the input image
        output_path: path of the output image"""
    lr = tf.image.decode_image(tf.io.read_file(image_path), channels=3)
    hr = tf.image.decode_image(tf.io.read_file(output_image_path), channels=3)

    lr.set_shape([None, None, 3])
    hr.set_shape([None, None, 3])

    lr = tf.image.convert_image_dtype(lr, tf.float32)
    hr = tf.image.convert_image_dtype(hr, tf.float32)

    lr = tf.image.resize(lr, [250, 250], method='bicubic')
    hr = tf.image.resize(hr, [250, 250])
    return lr, hr


Train_dataset = Training_images.map(load_image)
Valid_dataset = Validating_images.map(load_image)

dataset = Train_dataset.cache()
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(batch_size=32)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

val_dataset = Valid_dataset.batch(32).prefetch(tf.data.AUTOTUNE)


class EnhanceModel(Model):
    def __init__(self):
        super(EnhanceModel, self).__init__()
        self.conv_1 = Conv2D(128, (9, 9), padding='same', strides=1, activation='relu', name='first_conv')
        self.conv_2 = Conv2D(64, (3, 3), padding='same', strides=1, activation='relu', name='second_conv')
        self.conv_3 = Conv2D(3, (5, 5), padding='same', strides=1, activation='relu', name='third_conv')

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


# def psnr_metric(y_true, y_pred):
#     """PSNR metric from the SRCNN paper"""
#     return tf.image.psnr(y_true, y_pred, max_val=1.0)

#
# def ssim_metric(y_true, y_pred):
#     """SSIM metric from the SRCNN paper"""
#     return tf.image.ssim(y_true, y_pred, max_val=1.0)


model = EnhanceModel()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(dataset, validation_data=val_dataset, epochs=200, verbose=1)
model.save('enhance_model_v1.0.3.h5')
