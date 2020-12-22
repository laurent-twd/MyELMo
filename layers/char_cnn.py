import tensorflow as tf

class CharCNN(tf.keras.layers.Layer):

    def __init__(self, d_embeddings, n_chars, filters, activation):
        super(CharCNN, self).__init__()
        self.d_embeddings = d_embeddings
        self.n_chars = n_chars
        self.filters = filters
        self.embedding = tf.keras.layers.Embedding(self.n_chars, self.d_embeddings)
        self.cnn_layers = {}
        self.activation = activation
        for kernel_size, n_filters in self.filters.items():
            self.cnn_layers["size_{}".format(kernel_size)] = tf.keras.layers.Conv1D(
                filters = n_filters,
                kernel_size = kernel_size,
                strides=1,
                padding = 'valid')

    def call(self, inputs):

        embedding_inputs = self.embedding(inputs)
        
        char_embeddings = []
        for kernel_size, _ in self.filters.items():
            x = self.cnn_layers['size_{}'.format(kernel_size)](embedding_inputs)
            x = tf.reduce_max(x, axis = 2)
            x = self.activation(x)
            char_embeddings.append(x)
        char_embeddings = tf.concat(char_embeddings, axis =  2)

        return char_embeddings

        


