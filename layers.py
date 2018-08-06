import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Lambda


class WeightedSum(Layer):
    def build(self, input_shape):
        self.weight = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], 1),
                                      initializer='normal')
        super().build(input_shape)

    def call(self, x, mask=None):
        alpha = tf.nn.softmax(K.dot(x, self.weight), axis=1)
        return tf.squeeze(tf.matmul(alpha, x, transpose_a=True), axis=1)

    def compute_mask(self, x, mask=None):
        if mask is None:
            return None
        return mask

    def compute_output_shape(self, input_shape):
        batch, _, d = input_shape
        return (batch, d)


class WordInQuestionB(Layer):
    def call(self, inputs, mask=None):
        question, context = inputs
        question = tf.expand_dims(question, axis=1)
        context = tf.expand_dims(context, axis=2)
        return tf.expand_dims(tf.cast(tf.reduce_any(tf.equal(context, question), axis=2), tf.float32), axis=-1)

    def compute_output_shape(self, input_shape):
        batch, seq_len = input_shape[1]
        return (batch, seq_len, 1)

    def compute_mask(self, x, mask=None):
        if mask is None:
            return None
        return mask


class WordInQuestionW(Layer):
    def build(self, input_shape):
        self.weight = self.add_weight(name='kernel',
                                      shape=(input_shape[0][-1], 1),
                                      initializer='normal')
        super().build(input_shape)

    def call(self, inputs, mask=None):
        question, context = inputs
        question = tf.expand_dims(question, axis=1)
        context = tf.expand_dims(context, axis=2)
        similarity = tf.squeeze(K.dot(context * question, self.weight), axis=-1)
        return tf.expand_dims(tf.reduce_sum(tf.nn.softmax(similarity, axis=2), axis=2), axis=-1)

    def compute_output_shape(self, input_shape):
        batch, seq_len, d = input_shape[1]
        return (batch, seq_len, 1)

    def compute_mask(self, x, mask=None):
        if mask is None:
            return None
        return mask


class PositionPointer(Layer):
    def __init__(self, hidden_size, **kwargs):
        self.hidden_size = hidden_size
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.weight = self.add_weight(name='weight',
                                      shape=(1, input_shape[-1], self.hidden_size),
                                      initializer='normal')
        self.bias = self.add_weight(name='bias',
                                    shape=(self.hidden_size,),
                                    initializer='normal')
        self.v = self.add_weight(name='v',
                                 shape=(1, self.hidden_size, 1),
                                 initializer='normal')
        super().build(input_shape)

    def call(self, x, mask=None):
        pos = tf.nn.relu(K.bias_add(K.conv1d(x, self.weight), self.bias))
        logits = tf.squeeze(K.conv1d(pos, self.v), axis=-1)
        return tf.nn.softmax(logits, axis=-1)

    def compute_output_shape(self, input_shape):
        batch, seq_len, d = input_shape
        return (batch, seq_len)


class IndexSelect(Lambda):
    def __init__(self, **kwargs):
        def func(inputs):
            x, indices = inputs
            # (batch, seq_len, hidden_size)
            shape = x.shape.as_list()
            mask = tf.transpose(K.one_hot(tf.to_int32(indices), shape[1]), [0, 2, 1])
            # return tf.boolean_mask(x, mask)
            return tf.reduce_sum(x * mask, axis=1)

        super().__init__(function=func, **kwargs)

    def compute_output_shape(self, input_shape):
        batch, _, d = input_shape[0]
        return (batch, d)
