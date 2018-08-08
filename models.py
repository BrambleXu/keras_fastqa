import tensorflow as tf
from keras import Model
from keras.layers import Input, Embedding, LSTM, Conv1D, Concatenate, \
    RepeatVector, Multiply, Reshape, Lambda, Dropout

from layers import WeightedSum, WordInQuestionB, WordInQuestionW, PositionPointer, \
    IndexSelect, SequenceLength, Argmax, Backward


class FastQA:
    def __init__(self, vocab_size, embed_size, hidden_size,
                 question_limit=50, context_limit=400, dropout=.5,
                 pretrained_embeddings=None):
        embeddings = None
        if pretrained_embeddings is not None:
            embeddings = [pretrained_embeddings]
        self.embed_layer = Embedding(vocab_size, embed_size, weights=embeddings, trainable=False)
        self.q_lstm_f = LSTM(hidden_size, return_sequences=True)
        self.q_lstm_b = Backward(LSTM(hidden_size, return_sequences=True))
        self.c_lstm_f = LSTM(hidden_size, return_sequences=True)
        self.c_lstm_b = Backward(LSTM(hidden_size, return_sequences=True))
        self.q_fc = Conv1D(hidden_size, 1, activation='tanh', use_bias=False)
        self.c_fc = Conv1D(hidden_size, 1, activation='tanh', use_bias=False)
        self.start_pointer = PositionPointer(hidden_size)
        self.end_pointer = PositionPointer(hidden_size)
        self.q_limit = question_limit
        self.c_limit = context_limit
        self.hidden_size = hidden_size
        self.dropout = dropout

    def build(self):
        q_input = Input((self.q_limit,))
        c_input = Input((self.c_limit,))
        start_input = Input((1,))

        q_len = SequenceLength()(q_input)
        c_len = SequenceLength()(c_input)

        # question
        Q = self.embed_layer(q_input)
        Q_wiqb = WordInQuestionB()([q_input, q_input, q_len])
        Q_wiqw = WordInQuestionW()([Q, Q, q_len, q_len])
        Q_ = Concatenate()([Q_wiqb, Q_wiqw, Q])
        batch, _, feature_size = Q_.shape.as_list()
        Q_ = Dropout(self.dropout, (batch, 1, feature_size))(Q_)
        Z = Concatenate()([self.q_lstm_f(Q_), self.q_lstm_b([Q_, q_len])])
        Z = Reshape((self.q_limit, self.hidden_size * 2))(Z)
        Z = self.q_fc(Z)
        # context
        X = self.embed_layer(c_input)
        X_wiqb = WordInQuestionB()([q_input, c_input, c_len])
        X_wiqw = WordInQuestionW()([Q, X, q_len, c_len])
        X_ = Concatenate()([X_wiqb, X_wiqw, X])
        X_ = Dropout(self.dropout, (batch, 1, feature_size))(X_)
        H = Concatenate()([self.c_lstm_f(X_), self.c_lstm_b([X_, c_len])])
        H = Reshape((self.c_limit, self.hidden_size * 2))(H)
        H = self.c_fc(H)

        z = WeightedSum()([Z, c_len])
        # (batch, seq_eln, hidden_size)
        Z = RepeatVector(self.c_limit)(z)

        mul = Multiply()
        start_output = self.start_pointer(
            [Concatenate()([H, Z, mul([H, Z])]), c_len])
        h_s = IndexSelect()([H, start_input])
        H_s = RepeatVector(self.c_limit)(h_s)
        end_output = self.end_pointer(
            [Concatenate()([H, H_s, Z, mul([H, Z]), mul([H, H_s])]), c_len])

        # prediction
        start_index = Argmax()(start_output)
        h_s = IndexSelect()([H, start_index])
        H_s = RepeatVector(self.c_limit)(h_s)
        end_index = Argmax()(self.end_pointer(
            [Concatenate()([H, H_s, Z, mul([H, Z]), mul([H, H_s])]), c_len]))
        start_index = Lambda(lambda x: tf.squeeze(x, axis=-1))(start_index)
        end_index = Lambda(lambda x: tf.squeeze(x, axis=-1))(end_index)

        return Model(inputs=[q_input, c_input, start_input],
                     outputs=[start_output, end_output, start_index, end_index])
