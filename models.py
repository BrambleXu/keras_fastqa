from keras import backend as K
from keras import Model
from keras.layers import Input, Embedding, LSTM, Conv1D, Concatenate, \
    Bidirectional, RepeatVector, Multiply, Reshape

from layers import WeightedSum, WordInQuestionB, WordInQuestionW, PositionPointer, \
    IndexSelect, SequenceLength


class FastQA:
    def __init__(self, vocab_size, embed_size, hidden_size,
                 question_limit=50, context_limit=400):
        self.embed_layer = Embedding(vocab_size, embed_size)
        self.question_lstm = Bidirectional(LSTM(hidden_size, return_sequences=True))
        self.context_lstm = Bidirectional(LSTM(hidden_size, return_sequences=True))
        self.question_fc = Conv1D(hidden_size, 1, activation='tanh', use_bias=False)
        self.context_fc = Conv1D(hidden_size, 1, activation='tanh', use_bias=False)
        self.start_pointer = PositionPointer(hidden_size)
        self.end_pointer = PositionPointer(hidden_size)
        self.question_limit = question_limit
        self.context_limit = context_limit
        self.hidden_size = hidden_size

    def build(self):
        question_input = Input((self.question_limit,))
        context_input = Input((self.context_limit,))
        start_input = Input((1,))

        context_len = SequenceLength()(context_input)

        # question
        Q = self.embed_layer(question_input)
        Q_wiqb = WordInQuestionB()([question_input, question_input])
        Q_wiqw = WordInQuestionW()([Q, Q])
        Q_ = Concatenate()([Q_wiqb, Q_wiqw, Q])
        Z = self.question_lstm(Q_)
        Z = Reshape((self.question_limit, self.hidden_size * 2))(Z)
        Z = self.question_fc(Z)
        # context
        X = self.embed_layer(context_input)
        X_wiqb = WordInQuestionB()([question_input, context_input])
        X_wiqw = WordInQuestionW()([Q, X])
        X_ = Concatenate()([X_wiqb, X_wiqw, X])
        H = self.context_lstm(X_)
        H = Reshape((self.context_limit, self.hidden_size * 2))(H)
        H = self.context_fc(H)

        z = WeightedSum()([Z, context_len])
        context_limit = K.int_shape(context_input)[1]
        # (batch, seq_eln, hidden_size)
        Z = RepeatVector(self.context_limit)(z)

        mul = Multiply()
        start_output = self.start_pointer(
            [Concatenate()([H, Z, mul([H, Z])]), context_len])
        h_s = IndexSelect()([H, start_input])
        H_s = RepeatVector(context_limit)(h_s)
        end_output = self.end_pointer(
            [Concatenate()([H, H_s, Z, mul([H, Z]), mul([H, H_s])]), context_len])

        return Model(inputs=[question_input, context_input, start_input],
                     outputs=[start_output, end_output])
