from keras import backend as K
from keras import Model
from keras.layers import Input, Embedding, LSTM, Dense, Concatenate, \
    Bidirectional, RepeatVector

from layers import WeightedSum, WordInQuestionB, WordInQuestionW, PositionPointer, IndexSelect


class FastQA:
    def __init__(self, vocab_size, embed_size, hidden_size):
        self.embed_layer = Embedding(vocab_size, embed_size)
        self.question_lstm = Bidirectional(LSTM(hidden_size, return_sequences=True))
        self.context_lstm = Bidirectional(LSTM(hidden_size, return_sequences=True))
        self.question_fc = Dense(hidden_size)
        self.context_fc = Dense(hidden_size)
        self.start_pointer = PositionPointer(hidden_size)
        self.end_pointer = PositionPointer(hidden_size)

    def build(self):
        question_input = Input((None,))
        context_input = Input((None,))
        start_input = Input((None,))

        # question
        Q = self.embed_layer(question_input)
        Q_wiqb = WordInQuestionB()([question_input, question_input])
        Q_wiqw = WordInQuestionW()([Q, Q])
        Q = Concatenate()([Q_wiqb, Q_wiqw, Q])
        Z = self.question_lstm(Q)
        Z = self.question_fc(Z)
        # context
        X = self.embed_layer(context_input)
        X_wiqb = WordInQuestionB()([question_input, context_input])
        X_wiqw = WordInQuestionW()([Q, X])
        X = Concatenate()([X_wiqb, X_wiqw, X])
        H = self.context_lstm(X)
        H = self.context_fc(H)

        z = WeightedSum()(Z)
        context_len = K.int_shape(context_input)[1]
        # (batch, seq_eln, hidden_size)
        Z = RepeatVector(context_len)(z)

        start_output = self.start_pointer(Concatenate([H, Z, H * Z]))
        h_s = IndexSelect()([H, start_input])
        H_s = RepeatVector(context_len)(h_s)
        end_output = self.end_pointer(Concatenate([H, H_s, H * Z, H * H_s]))

        return Model(inputs=[question_input, context_input, start_input],
                     outputs=[start_output, end_output])
