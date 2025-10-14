from matplotlib import pyplot as plt
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('enhance_model_v1.0.2.h5',  compile=False)
image_file = 'Valid_x/0813x4m.png'


def procrss_image(path, model):
    img = tf.image.decode_image(tf.io.read_file(path), channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, axis=0)
    img = tf.image.resize(img, [215, 215])
    enhanced = model.predict(img)
    enhanced = enhanced[0]
    return enhanced


new_image = procrss_image(image_file, model)
print(new_image.shape)
# new_image_uint8 = np.clip(new_image * 255, 0, 255).astype(np.uint8)

# Convert RGB to BGR for OpenCV
new_image_bgr = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

cv2.imshow("Enhanced Image", new_image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
# old_image = cv2.imread(image_file)
# img1 = cv2.cvtColor(old_image, cv2.COLOR_BGR2RGB)
# img1 = cv2.resize(img1, (215, 215)) / 255.0

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(img1)
# plt.title('Blur Image')
# plt.axis('off')
#
#
# plt.subplot(1, 2, 2)
# plt.imshow(new_image)
# plt.title('Enhanced Image')
# plt.axis('off')