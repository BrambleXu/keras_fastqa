import tensorflow as tf
from keras import backend as K
from keras import Model
from keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Lambda, \
    Bidirectional

from layers import WeightedSum, WordInQuestionB, WordInQuestionW


class FastQA:
    def __init__(self, vocab_size, embed_size, hidden_size):
        self.embed_layer = Embedding(vocab_size, embed_size)
        self.question_lstm = Bidirectional(LSTM(hidden_size, return_sequences=True))
        self.context_lstm = Bidirectional(LSTM(hidden_size, return_sequences=True))
        self.question_fc = Dense(hidden_size)
        self.context_fc = Dense(hidden_size)
        self.answer_start_fc = Dense(hidden_size)
        self.start_fc = Dense(1)
        self.answer_end_fc = Dense(hidden_size)
        self.end_fc = Dense(1)

    def build(self):
        question_input = Input((None,))
        context_input = Input((None,))
        start_input = Input((None,))

        Q = self.embed_layer(question_input)
        X = self.embed_layer(context_input)
        X_wiqb = WordInQuestionB()([question_input, context_input])
        X_wiqw = WordInQuestionW()([Q, X])
        X = Concatenate()([X_wiqb, X_wiqw, X])
        Q_wiqb = WordInQuestionB()([question_input, question_input])
        Q_wiqw = WordInQuestionW()([Q, Q])
        Q = Concatenate()([Q_wiqb, Q_wiqw, Q])

        Z = self.question_lstm(Q)
        Z = self.question_fc(Z)
        z = WeightedSum()(Z)
        context_len = K.int_shape(context_input)[1]
        Z = Lambda(lambda x: tf.tile(x, [1, context_len, 1]))(z)

        H_dash = self.context_lstm(X)
        H = self.context_fc(H_dash)

        start = self.start_fc(self.answer_start_fc(Concatenate()([H, Z, H * Z])))
        end = self.end_fc(self.answer_end_fc(Concatenate()([H, Z, H * Z])))

        return Model(inputs=[question_input, context_input, start_input], outputs=[start, end])
