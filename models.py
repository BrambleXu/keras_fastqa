from keras import Model
from keras.layers import Input, Embedding, LSTM, Conv1D, Concatenate, \
    RepeatVector, Multiply, Reshape, Dropout

from layers import WeightedSum, WordInQuestionB, WordInQuestionW, \
    IndexSelect, SequenceLength, Argmax, Backward, Highway, Ones, \
    StartPointer, EndPointer
from initializers import init_lstm_projection


class FastQA:
    def __init__(self, vocab_size, embed_size, hidden_size,
                 question_limit=50, context_limit=400, dropout=.5,
                 pretrained_embeddings=None, with_feature=True):
        embeddings = None
        if pretrained_embeddings is not None:
            embeddings = [pretrained_embeddings]
        self.embed_layer = Embedding(vocab_size, embed_size, weights=embeddings, trainable=False)
        self.highway = Highway(hidden_size)
        self.lstm_f = LSTM(hidden_size, return_sequences=True)
        self.lstm_b = Backward(LSTM(hidden_size, return_sequences=True))
        self.q_fc = Conv1D(hidden_size, 1, activation='tanh', use_bias=False,
                           kernel_initializer=init_lstm_projection)
        self.c_fc = Conv1D(hidden_size, 1, activation='tanh', use_bias=False,
                           kernel_initializer=init_lstm_projection)
        self.start_pointer = StartPointer(hidden_size)
        self.end_pointer = EndPointer(hidden_size)
        self.q_limit = question_limit
        self.c_limit = context_limit
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.with_feature = with_feature

    def build(self):
        q_input = Input((self.q_limit,))
        c_input = Input((self.c_limit,))
        start_input = Input((1,))

        batch = q_input.shape.as_list()[0]
        q_len = SequenceLength()(q_input)
        c_len = SequenceLength()(c_input)

        # question
        # embedding
        Q_e = self.embed_layer(q_input)
        # feature
        if self.with_feature:
            Q_wiq = Ones(2)([q_input, q_len])
        # embedding
        Q_h = self.highway(Q_e)
        Q_d = Dropout(self.dropout, (batch, 1, self.embed_size))(Q_h)
        # lstm input
        if self.with_feature:
            Q_ = Concatenate()([Q_d, Q_wiq])
        else:
            Q_ = Q_d
        # bilstm
        Z = Concatenate()([self.lstm_f(Q_), self.lstm_b([Q_, q_len])])
        Z = Reshape((self.q_limit, self.hidden_size * 2))(Z)
        # projection
        Z = self.q_fc(Z)

        # context
        # embedding
        X_e = self.embed_layer(c_input)
        # feature
        if self.with_feature:
            X_wiqb = WordInQuestionB()([q_input, c_input, c_len])
            X_wiqw = WordInQuestionW()([Q_e, X_e, q_len, c_len])
        # embedding
        X_h = self.highway(X_e)
        X_d = Dropout(self.dropout, (batch, 1, self.embed_size))(X_h)
        # lstm input
        if self.with_feature:
            X_ = Concatenate()([X_d, X_wiqb, X_wiqw])
        else:
            X_ = X_d
        # bilstm
        H = Concatenate()([self.lstm_f(X_), self.lstm_b([X_, c_len])])
        H = Reshape((self.c_limit, self.hidden_size * 2))(H)
        # projection
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
            [Concatenate()([H, H_s, Z, mul([H, Z]), mul([H, H_s])]), c_len, start_input])

        # prediction
        start_index = Argmax()(start_output)
        h_s = IndexSelect()([H, start_index])
        H_s = RepeatVector(self.c_limit)(h_s)
        end_index = Argmax()(self.end_pointer(
            [Concatenate()([H, H_s, Z, mul([H, Z]), mul([H, H_s])]), c_len, start_index]))

        return Model(inputs=[q_input, c_input, start_input],
                     outputs=[start_output, end_output, start_index, end_index])
