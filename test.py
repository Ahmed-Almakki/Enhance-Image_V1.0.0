import tensorflow as tf
import cv2

# 2. Load trained weights
model = tf.keras.models.load_model('enhance_model_v1.0.3.h5',
                                   custom_objects={
                                       'mse': tf.keras.losses.MeanSquaredError(),
                                       'mae': tf.keras.metrics.MeanAbsoluteError()
                                   }
                                   )

# 3. Load and preprocess the image

image_file = 'Valid_x/0801x4m.png'
kk = cv2.imread(image_file)
cv2.imshow('asd', kk)
img = tf.image.decode_image(tf.io.read_file(image_file), channels=3)
img = tf.image.convert_image_dtype(img, tf.float32)
img = tf.expand_dims(img, axis=0)

# 4. Predict
enhanced = model.predict(img)[0]
enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
enhanced = tf.cast(tf.clip_by_value(enhanced * 255.0, 0, 255), tf.uint8)
print(f'img size{enhanced.shape}\t{type(enhanced)}')
xx = cv2.imwrite('add.png', enhanced.numpy())
cv2.imshow("Enhanced Image", enhanced.numpy())
cv2.waitKey(0)
cv2.destroyAllWindows()
