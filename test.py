from keras.src.utils import img_to_array, load_img
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import PIL
from Model_Architecture import DepthToSpace

model = tf.keras.models.load_model('Enhancev_5.keras', compile=False, custom_objects={'DepthToSpace': DepthToSpace})
image_path = 'Valid_x/0801x4m.png'


def getLowerRes(img, factor=3):
    """
    Resize the low resolution image because the model is scaling up the image by 3
    hence the default factor is 3
    :param img: low resolution image
    :param factor: resize the image using this factor by default 3
    :return: scale down image
    """
    return img.resize((img.size[0] // factor, img.size[1] // factor), PIL.Image.BICUBIC)


def getUpperRes(model, img):
    """
    Enhance the image and preprocess it after and before enhancing, so it works well
    with the model
    :param model: responsible for enhancing
    :param img: low resolution image
    :return: enhanced image
    """
    img = img.convert('YCbCr')
    y, cb, cr = img.split()

    img_y = img_to_array(y)
    img_y = img_y.astype('float32') / 255.0
    img_y = np.expand_dims(img_y, axis=0)

    pred = model.predict(img_y)
    pred = pred[0]
    pred *= 255.0
    pred = pred.clip(0, 255)
    pred = pred.reshape((np.shape(pred)[0], np.shape(pred)[1]))

    out_y = PIL.Image.fromarray(np.uint8(pred), mode='L')
    out_cb = cb.resize(out_y.size, PIL.Image.BICUBIC)
    out_cr = cr.resize(out_y.size, PIL.Image.BICUBIC)

    out_img = PIL.Image.merge('YCbCr', (out_y, out_cb, out_cr)).convert('RGB')
    return out_img


def plot_result(in_img, out_img):
    """Plot two images
    :arg
    in_img: the original low resolution image
    out_img: the enhanced image
    """
    arr_in_img = img_to_array(in_img)
    arr_out_img = img_to_array(out_img)

    arr_in_img = arr_in_img.astype('float32') / 255.0
    arr_out_img = arr_out_img.astype('float32') / 255.0

    plt.subplot(1, 2, 1)
    plt.imshow(arr_in_img)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(arr_out_img)
    plt.title('Enhanced Image')

    plt.show()


input_image = load_img(image_path)
lower_image = getLowerRes(input_image, 1)
enhance_image = getUpperRes(model, lower_image)

# Display the Enhanced Image
enhance_image.show()

# Display Both images the enhanced and the original image
# plot_result(input_image, enhance_image)

