import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, learning_rate = 5e-4, num_train_steps = 500000, warmup_steps = 10000):
    super(CustomSchedule, self).__init__()
    
    self.warmup_steps = tf.cast(warmup_steps, dtype = tf.float32)
    self.learning_rate = learning_rate
    self.num_train_steps = tf.cast(num_train_steps, dtype = tf.float32)
    
  def __call__(self, step):
    step = tf.minimum(step, self.num_train_steps)
    learning_rate = ((self.learning_rate - 0.) * (1 - step / self.num_train_steps)) + 0.
    learning_rate *= tf.minimum(1.0, tf.cast(step, tf.float32) / self.warmup_steps)
    return learning_rate * 32. / 128.

