"""Enhance Image Using Tensorflow"""
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

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
    """Load the image to create the pipline
    :param:
        image_path: the path of the input image
        output_path: path of the output image"""
    print(f"first {image_path}\t{output_image_path}")
    lr = tf.image.decode_image(tf.io.read_file(image_path), channels=3)
    hr = tf.image.decode_image(tf.io.read_file(output_image_path), channels=3)

    lr.set_shape([None, None, 3])
    hr.set_shape([None, None, 3])

    print(f"second {lr}\n{hr}")
    lr = tf.image.convert_image_dtype(lr, tf.float32)
    hr = tf.image.convert_image_dtype(hr, tf.float32)

    print(f"\n-------------------------------------\nthird{lr}\n{hr}")
    lr = tf.image.resize(lr, [215, 215])
    hr = tf.image.resize(hr, [215, 215])
    return lr, hr


Train_dataset = Training_images.map(load_image)
Valid_dataset = Validating_images.map(load_image)
# print(f"train_img: {train_img}\nTraining_image: {Training_images}\nTrainDataset: {Train_dataset}")
# for x, y in Train_dataset:
#     print(f"shap: {x.shape}\n x: {x}\n y: {y}")
#     plt.imshow(y, interpolation='nearest')
#     plt.show()
#     break

dataset = Train_dataset.cache()
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(batch_size=32)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


class Enhance_Model:
    def __init__(self):