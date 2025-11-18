import tensorflow as tf


class DepthToSpace(tf.keras.layers.Layer):
    def __init__(self, block_size, **kwargs):
        super(DepthToSpace, self).__init__(**kwargs)
        self.block_size = block_size

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.block_size)

    def get_config(self):
        config = super(DepthToSpace, self).get_config()
        config.update({"block_size": self.block_size})
        return config