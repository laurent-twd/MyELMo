import tensorflow as tf

class HighwayLayer(tf.keras.layers.Layer):

    def __init__(self, input_dim, activation):
        super(HighwayLayer, self).__init__()
        self.input_dim = input_dim
        self.activation = activation
        self.dense_gate = tf.keras.layers.Dense(input_dim, activation = 'linear')
        self.dense_transform = tf.keras.layers.Dense(input_dim, activation = 'linear')

    def call(self, x):
        gate = tf.math.sigmoid(self.dense_gate(x))
        transform = self.activation(self.dense_transform(x))
        return x * gate + (1. - gate) * transform
