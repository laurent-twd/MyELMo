import collections
from absl import logging
import tensorflow as tf

from layers.highway_layer import HighwayLayer
from layers.char_cnn import CharCNN

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def gelu(features, approximate=False, name=None):

  with ops.name_scope(name, "Gelu", [features]):
    features = ops.convert_to_tensor(features, name="features")
    if approximate:
      coeff = math_ops.cast(0.044715, features.dtype)
      return 0.5 * features * (
          1.0 + math_ops.tanh(0.7978845608028654 *
                              (features + coeff * math_ops.pow(features, 3))))
    else:
      return 0.5 * features * (1.0 + math_ops.erf(
          features / math_ops.cast(1.4142135623730951, features.dtype)))

class ELMoEncoder(tf.keras.Model):

  def __init__(
      self,
      vocab_size,
      n_chars,
      filters,
      output_dim,
      d_embeddings = 64,
      hidden_size=256,
      num_layers=2,
      num_highway_layers = 2,
      inner_activation=lambda x: gelu(x, approximate=True),
      output_dropout=0.5,
      recurrent_dropout=0.5,
      initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
      output_range=None,
      embedding_width=None,
      embedding_layer=None,
      **kwargs):

    super(ELMoEncoder, self).__init__()
    
    self.activation = tf.keras.activations.get(inner_activation)
    self.initializer = tf.keras.initializers.get(initializer)
    self.num_highway_layers = num_highway_layers
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.output_dim =  output_dim

    self.charCNN = CharCNN(d_embeddings, n_chars, filters, lambda x: gelu(x, approximate=True))
    self.size_output_charCNN = sum([k for k in self.charCNN.filters.values()])
    self.highway_layers = [HighwayLayer(input_dim=self.size_output_charCNN, 
                                  activation=lambda x: gelu(x, approximate=True)) for _ in range(num_highway_layers)]
    self.projection = tf.keras.layers.Dense(2 * self.hidden_size)

    self.embedding_norm_layer = tf.keras.layers.LayerNormalization(
        name='embeddings/layer_norm', axis=-1, epsilon=1e-12, dtype=tf.float32)
    self.dropout = tf.keras.layers.Dropout(rate=output_dropout)

    self.lstm_layers = [tf.keras.layers.LSTM(hidden_size, dropout = output_dropout, recurrent_dropout = recurrent_dropout, return_sequences = True, return_state = False) 
                  for _ in range(2 * num_layers)] 

    self.dense_logits = tf.keras.layers.Dense(self.output_dim, activation = 'linear')

    self.pooler_layer = tf.keras.layers.Dense(
        units=self.hidden_size,
        activation='tanh',
        kernel_initializer=self.initializer,
        name='pooler_transform')

  def call(self, char_ids, mask, training): 

    word_embeddings = self.charCNN(char_ids)
    for i in range(self.num_highway_layers):
      word_embeddings = self.highway_layers[i](word_embeddings)
    word_embeddings = self.projection(word_embeddings)

    embeddings = self.embedding_norm_layer(word_embeddings)
    forward_embeddings = self.dropout(embeddings, training = training)

    sequence_lengths = tf.cast(tf.reduce_sum(1. - mask, axis = 1), dtype = tf.int32)
    backward_embeddings = tf.reverse_sequence(forward_embeddings, seq_lengths = sequence_lengths, seq_axis = 1)

    encoder_outputs = [forward_embeddings]
    for i in range(self.num_layers):
      forward_embeddings = self.lstm_layers[i](forward_embeddings, training = training)
      backward_embeddings = self.lstm_layers[i + self.num_layers](backward_embeddings, training = training)
      layer_embedding = tf.concat([forward_embeddings, tf.reverse_sequence(backward_embeddings, seq_lengths = sequence_lengths, seq_axis = 1)], axis = 2)      
      encoder_outputs.append(layer_embedding)

    last_encoder_output = encoder_outputs[-1]
    first_token_tensor = last_encoder_output[:, 0, :]
    cls_output = self.pooler_layer(first_token_tensor)

    forward_last_encoder_output, backward_last_encoder_output = tf.split(last_encoder_output, num_or_size_splits = 2, axis = 2)
    forward_logits = self.dense_logits(forward_last_encoder_output)
    backward_logits = self.dense_logits(backward_last_encoder_output)

    outputs = dict(
        sequence_output=encoder_outputs[-1],
        pooled_output=cls_output,
        encoder_outputs=encoder_outputs,
        forward_logits=forward_logits,
        backward_logits=backward_logits
    )

    return outputs