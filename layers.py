import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer


class WeightedSum(Layer):
    def build(self, input_shape):
        self.weight = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], 1),
                                      initializer='normal')

    def call(self, x):
        alpha = tf.nn.softmax(K.dot(x, self.weight), axis=1)
        return tf.matmul(alpha, x, transpose_a=True)


class WordInQuestionB(Layer):
    def call(self, inputs):
        question, context = inputs
        question = tf.expand_dims(question, dim=1)
        context = tf.expand_dims(context, dim=2)
        return tf.expand_dims(tf.cast(tf.reduce_any(tf.equal(context, question), axis=2), tf.float32), dim=-1)


class WordInQuestionW(Layer):
    def build(self, input_shape):
        self.weight = self.add_weight(name='kernel',
                                      shape=(input_shape[0][-1], 1),
                                      initializer='normal')

    def call(self, inputs):
        question, context = inputs
        question = tf.expand_dims(question, dim=1)
        context = tf.expand_dims(context, dim=2)
        similarity = tf.squeeze(K.dot(context * question, self.weight), axis=-1)
        return tf.expand_dims(tf.reduce_sum(tf.nn.softmax(similarity, axis=2), axis=2), dim=-1)
