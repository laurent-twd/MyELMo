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
    activation = tf.keras.activations.get(inner_activation)
    initializer = tf.keras.initializers.get(initializer)

    char_ids = tf.keras.layers.Input(
        shape=(None, None), dtype=tf.int32, name='input_char_ids')
    mask = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_mask')

    charCNN = CharCNN(d_embeddings, n_chars, filters, lambda x: gelu(x, approximate=True))
    size_output_charCNN = sum([k for k in charCNN.filters.values()])

    highway_layers = [HighwayLayer(input_dim=size_output_charCNN, 
                                  activation=lambda x: gelu(x, approximate=True)) for _ in range(num_highway_layers)]

    word_embeddings = charCNN(char_ids)
    for i in range(num_highway_layers):
      word_embeddings = highway_layers[i](word_embeddings)

    word_embeddings = tf.keras.layers.Dense(2 * hidden_size)(word_embeddings)
    embedding_norm_layer = tf.keras.layers.LayerNormalization(
        name='embeddings/layer_norm', axis=-1, epsilon=1e-12, dtype=tf.float32)

    embeddings = embedding_norm_layer(word_embeddings)
    forward_embeddings = (tf.keras.layers.Dropout(rate=output_dropout)(embeddings))

    sequence_lengths = tf.reduce_sum(1 - mask, axis = 1)
    backward_embeddings = forward_embeddings
    #backward_embeddings = tf.reverse_sequence(forward_embeddings, seq_lengths = sequence_lengths, seq_axis = 1)

    lstm_layers = [tf.keras.layers.LSTM(hidden_size, dropout = output_dropout, recurrent_dropout = recurrent_dropout, return_sequences = True, return_state = False) 
                  for _ in range(2 * num_layers)]    

    
    encoder_outputs = [forward_embeddings]
    for i in range(num_layers):
      forward_embeddings = lstm_layers[i](forward_embeddings)
      backward_embeddings = lstm_layers[i + num_layers](backward_embeddings)
      #layer_embedding = tf.concat([forward_embeddings, tf.reverse_sequence(backward_embeddings, seq_lengths = sequence_lengths, seq_axis = 1)], axis = 2)
      layer_embedding = tf.concat([forward_embeddings, backward_embeddings], axis = 2)
      
      encoder_outputs.append(layer_embedding)

    last_encoder_output = encoder_outputs[-1]
    first_token_tensor = last_encoder_output[:, 0, :]
    pooler_layer = tf.keras.layers.Dense(
        units=hidden_size,
        activation='tanh',
        kernel_initializer=initializer,
        name='pooler_transform')
    cls_output = pooler_layer(first_token_tensor)

    dense_logits = tf.keras.layers.Dense(output_dim, activation = 'linear')

    forward_last_encoder_output, backward_last_encoder_output = tf.split(last_encoder_output, num_or_size_splits = 2, axis = 2)
    forward_logits = dense_logits(forward_last_encoder_output)
    backward_logits = dense_logits(backward_last_encoder_output)

    outputs = dict(
        sequence_output=encoder_outputs[-1],
        pooled_output=cls_output,
        encoder_outputs=encoder_outputs,
        forward_logits=forward_logits,
        backward_logits=backward_logits
    )

    super(ELMoEncoder, self).__init__(
        inputs=[char_ids, mask], outputs=outputs, **kwargs)

    config_dict = {
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'inner_activation': tf.keras.activations.serialize(activation),
        'output_dropout': output_dropout,
        'recurrent_dropout': recurrent_dropout,
        'initializer': tf.keras.initializers.serialize(initializer),
        'output_range': output_range,
        'embedding_width': embedding_width,
        'embedding_layer': embedding_layer,
    }

    config_cls = collections.namedtuple('Config', config_dict.keys())
    self._config = config_cls(**config_dict)
    self._pooler_layer = pooler_layer
    self._embedding_norm_layer = embedding_norm_layer

  def get_config(self):
    return dict(self._config._asdict())

  @property
  def pooler_layer(self):
    """The pooler dense layer after the transformer layers."""
    return self._pooler_layer

  @classmethod
  def from_config(cls, config, custom_objects=None):
    if 'embedding_layer' in config and config['embedding_layer'] is not None:
      warn_string = (
          'You are reloading a model that was saved with a '
          'potentially-shared embedding layer object. If you contine to '
          'train this model, the embedding layer will no longer be shared. '
          'To work around this, load the model outside of the Keras API.')
      print('WARNING: ' + warn_string)
      logging.warn(warn_string)

    return cls(**config)
