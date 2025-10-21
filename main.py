# """Enhance Image Using Tensorflow, Fine-Tuning SRCNN model and also using RGB image
#     instead of GreyScale image"""
# import tensorflow as tf
# from keras.layers import Input, Conv2D, UpSampling2D
# from keras.models import Model
#
# train_path = 'Train_x'
# output_path = 'Train_Y'
# valid_path = 'Valid_x'
# valid_output_path = 'Valid_Y'
#
# train_img = tf.data.Dataset.list_files(train_path + '/*.*', shuffle=False)
# output_img = tf.data.Dataset.list_files(output_path + '/*.*', shuffle=False)
# valid_img = tf.data.Dataset.list_files(valid_path + '/*.*', shuffle=False)
# valid_output = tf.data.Dataset.list_files(valid_output_path + '/*.*', shuffle=False)
#
# Training_images = tf.data.Dataset.zip((train_img, output_img))
# Validating_images = tf.data.Dataset.zip((valid_img, valid_output))
#
#
# def load_image(image_path, output_image_path):
#     """Load the image to create the pipline and process the image by resizing and normalizing
#     :param:
#         image_path: the path of the input image
#         output_path: path of the output image"""
#     lr = tf.image.decode_image(tf.io.read_file(image_path), channels=3)
#     hr = tf.image.decode_image(tf.io.read_file(output_image_path), channels=3)
#
#     lr.set_shape([None, None, 3])
#     hr.set_shape([None, None, 3])
#
#     lr = tf.image.convert_image_dtype(lr, tf.float32)
#     hr = tf.image.convert_image_dtype(hr, tf.float32)
#
#     return lr, hr
#
#
# Train_dataset = Training_images.map(load_image)
# Valid_dataset = Validating_images.map(load_image)
#
# dataset = Train_dataset.cache()
# dataset = dataset.shuffle(buffer_size=1000)
# dataset = dataset.batch(batch_size=1)
# dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
#
# val_dataset = Valid_dataset.batch(1).prefetch(tf.data.AUTOTUNE)
#
#
# def enhance_model():
#     input = Input(shape=(None, None, 3))
#     x = Conv2D(128, (9, 9), padding='same', strides=1, activation='relu')(input)
#     x = Conv2D(64, (3, 3), padding='same', strides=1, activation='relu')(x)
#     x = Conv2D(3, (5, 5), padding='same', strides=1, activation='relu')(x)
#     x = UpSampling2D((4, 4), interpolation='bicubic')(x)
#     return Model(inputs=input, outputs=x)
#
#
# # def psnr_metric(y_true, y_pred):
# #     """PSNR metric from the SRCNN paper"""
# #     return tf.image.psnr(y_true, y_pred, max_val=1.0)
#
# #
# # def ssim_metric(y_true, y_pred):
# #     """SSIM metric from the SRCNN paper"""
# #     return tf.image.ssim(y_true, y_pred, max_val=1.0)
# model = enhance_model()
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# model.fit(dataset, validation_data=val_dataset, epochs=10, verbose=1)
# model.save('enhance_model_v1.0.3.h5')

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load the model
model = tf.saved_model.load('saved_model')
infer = model.signatures["serving_default"]

# Load and preprocess the image
image_path = 'Train_x/0001x4m.png'
x = tf.image.decode_image(tf.io.read_file(image_path), channels=3)  # ensure RGB
x = tf.cast(x, tf.float32) / 255.0  # normalize if model expects 0-1
x_input = tf.expand_dims(x, 0)      # add batch dimension

# Predict
output = infer(x_input)
print(f'first {output}')
# Extract predicted tensor (assuming single output)
# output_tensor = list(output.values())[0]

