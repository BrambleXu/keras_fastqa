import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Lambda, Wrapper


class SequenceLength(Lambda):
    def __init__(self, **kwargs):
        def func(x):
            mask = tf.cast(x, tf.bool)
            length = tf.reduce_sum(tf.to_int32(mask), axis=1)
            return length

        super().__init__(function=func, **kwargs)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size,)


class WeightedSum(Layer):
    def build(self, input_shape):
        self.weight = self.add_weight(name='kernel',
                                      shape=(input_shape[0][-1], 1),
                                      initializer='ones')
        super().build(input_shape)

    def call(self, inputs):
        x, seq_len = inputs
        # (batch, seq_len, 1)
        logits = K.dot(x, self.weight)
        mask = tf.expand_dims(
            tf.sequence_mask(seq_len, maxlen=x.shape.as_list()[1], dtype=tf.float32),
            axis=-1)
        logits = logits + tf.float32.min * (1 - mask)
        alpha = tf.nn.softmax(logits, axis=1)
        return tf.squeeze(tf.matmul(alpha, x, transpose_a=True), axis=1)

    def compute_output_shape(self, input_shape):
        batch, _, d = input_shape[0]
        return (batch, d)


class Ones(Lambda):
    def __init__(self, output_size, **kwargs):
        self.output_size = output_size

        def func(inputs):
            q, q_len = inputs
            _, seq_len = q.shape.as_list()
            x = tf.ones((1, seq_len), dtype=tf.float32)
            mask = tf.sequence_mask(q_len, maxlen=seq_len, dtype=tf.float32)
            x = tf.expand_dims(x * mask, axis=-1)
            return tf.tile(x, [1, 1, output_size])

        super().__init__(function=func, **kwargs)

    def compute_output_shape(self, input_shape):
        batch, seq_len = input_shape[0]
        return (batch, seq_len, self.output_size)


class WordInQuestionB(Lambda):
    def __init__(self, **kwargs):
        def func(inputs):
            question, context, context_len = inputs
            question = tf.expand_dims(question, axis=1)
            context = tf.expand_dims(context, axis=2)
            wiq_b = tf.to_float(tf.reduce_any(tf.equal(context, question), axis=2))
            mask = tf.sequence_mask(context_len, maxlen=context.shape.as_list()[1], dtype=tf.float32)
            return tf.expand_dims(wiq_b * mask, axis=-1)

        super().__init__(function=func, **kwargs)

    def compute_output_shape(self, input_shape):
        batch, seq_len = input_shape[1]
        return (batch, seq_len, 1)


class WordInQuestionW(Layer):
    def build(self, input_shape):
        self.weight = self.add_weight(name='kernel',
                                      shape=(input_shape[0][-1], 1),
                                      initializer='ones')
        super().build(input_shape)

    def call(self, inputs):
        question, context, question_len, context_len = inputs
        question = tf.expand_dims(question, axis=1)
        context = tf.expand_dims(context, axis=2)
        similarity = tf.squeeze(K.dot(context * question, self.weight), axis=-1)
        question_mask = tf.expand_dims(tf.sequence_mask(
            question_len, maxlen=question.shape.as_list()[2], dtype=tf.float32), axis=1)
        context_mask = tf.expand_dims(tf.sequence_mask(
            context_len, maxlen=context.shape.as_list()[1], dtype=tf.float32), axis=2)
        mask = tf.matmul(context_mask, question_mask)
        similarity = similarity + tf.float32.min * (1 - mask)
        return tf.expand_dims(tf.reduce_sum(tf.nn.softmax(similarity, axis=1) * mask, axis=2), axis=-1)

    def compute_output_shape(self, input_shape):
        batch, seq_len, d = input_shape[1]
        return (batch, seq_len, 1)


class Highway(Layer):
    def __init__(self, hidden_size, **kwargs):
        self.hidden_size = hidden_size
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.projection = self.add_weight(name='projection',
                                          shape=(1, input_shape[-1], self.hidden_size),
                                          initializer='glorot_uniform')
        self.W_h = self.add_weight(name='W_h',
                                   shape=(1, self.hidden_size, self.hidden_size),
                                   initializer='glorot_uniform')
        self.b_h = self.add_weight(name='b_h',
                                   shape=(self.hidden_size,),
                                   initializer='zeros')
        self.W_t = self.add_weight(name='W_t',
                                   shape=(1, self.hidden_size, self.hidden_size),
                                   initializer='glorot_uniform')
        self.b_t = self.add_weight(name='b_t',
                                   shape=(self.hidden_size,),
                                   initializer='zeros')

    def call(self, x):
        x = K.conv1d(x, self.projection)
        H = tf.nn.tanh(K.bias_add(K.conv1d(x, self.W_h), self.b_h))
        T = tf.nn.sigmoid(K.bias_add(K.conv1d(x, self.W_t), self.b_t))
        return T * x + (1 - T) * H

    def compute_output_shape(self, input_shape):
        batch, seq_len, d = input_shape
        return (batch, seq_len, self.hidden_size)


class StartPointer(Layer):
    def __init__(self, hidden_size, **kwargs):
        self.hidden_size = hidden_size
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.weight = self.add_weight(name='weight',
                                      shape=(1, input_shape[0][-1], self.hidden_size),
                                      initializer='glorot_uniform')
        self.bias = self.add_weight(name='bias',
                                    shape=(self.hidden_size,),
                                    initializer='zeros')
        self.v = self.add_weight(name='v',
                                 shape=(1, self.hidden_size, 1),
                                 initializer='glorot_uniform')
        super().build(input_shape)

    def call(self, inputs):
        x, seq_len = inputs
        pos = tf.nn.relu(K.bias_add(K.conv1d(x, self.weight), self.bias))
        logits = tf.squeeze(K.conv1d(pos, self.v), axis=-1)
        mask = tf.sequence_mask(seq_len, maxlen=x.shape.as_list()[1], dtype=tf.float32)
        logits = logits + tf.float32.min * (1 - mask)
        return tf.nn.softmax(logits, axis=-1)

    def compute_output_shape(self, input_shape):
        batch, seq_len, d = input_shape[0]
        return (batch, seq_len)


class EndPointer(StartPointer):
    def call(self, inputs):
        x, seq_len, start_indices = inputs
        pos = tf.nn.relu(K.bias_add(K.conv1d(x, self.weight), self.bias))
        logits = tf.squeeze(K.conv1d(pos, self.v), axis=-1)
        maxlen = x.shape.as_list()[1]
        start_indices = tf.reshape(start_indices, [-1])
        left_mask = tf.sequence_mask(start_indices - 1, maxlen=maxlen, dtype=tf.float32)
        logits = logits + tf.float32.min * left_mask
        mask = tf.sequence_mask(seq_len, maxlen=maxlen, dtype=tf.float32)
        logits = logits + tf.float32.min * (1 - mask)
        return tf.nn.softmax(logits, axis=-1)


class IndexSelect(Lambda):
    def __init__(self, **kwargs):
        def func(inputs):
            x, indices = inputs
            # (batch, seq_len, hidden_size)
            indices = tf.reshape(tf.to_int32(indices), [-1])
            indices = tf.stack([tf.range(tf.shape(x)[0]), indices], axis=1)
            return tf.gather_nd(x, indices)

        super().__init__(function=func, **kwargs)

    def compute_output_shape(self, input_shape):
        batch, _, d = input_shape[0]
        return (batch, d)


class Argmax(Lambda):
    def __init__(self, axis=-1, **kwargs):
        def func(x):
            return tf.argmax(x, axis=axis, output_type=tf.int32)

        super().__init__(function=func, **kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],)


class Backward(Wrapper):
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)

    def build(self, input_shape):
        self.layer.build(input_shape)
        super().build()

    def call(self, inputs):
        x, seq_len = inputs
        x = tf.reverse_sequence(x, seq_len, seq_axis=1, batch_axis=0)
        x = self.layer.call(x, mask=None, training=None, initial_state=None)
        x = tf.reverse_sequence(x, seq_len, seq_axis=1, batch_axis=0)
        return x

    def compute_output_shape(self, input_shape):
        batch, seq_len, _ = input_shape[0]
        return (batch, seq_len, self.layer.units)
