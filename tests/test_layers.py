from unittest import TestCase

import numpy as np
from keras import Model
from keras.layers import Input, Embedding

from layers import WordInQuestionB, WordInQuestionW, SequenceLength


class WIQLayerTest(TestCase):
    def test_wiqb(self):
        # inputs
        q = np.array([[1, 2, 3, 0, 0, 0]], dtype=np.int32)
        c = np.array([[1, 4, 2, 5, 3, 6, 0, 0, 0, 0]], dtype=np.int32)

        # building graph
        q_input = Input((6,))
        c_input = Input((10,))
        c_len = SequenceLength()(c_input)
        output = WordInQuestionB()([q_input, c_input, c_len])
        model = Model([q_input, c_input], output)

        # assertion
        result = model.predict([q, c])
        expected = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)[None, :, None]
        np.testing.assert_array_equal(result, expected)

    def test_wiqw(self):
        # inputs
        q = np.array([[1, 2, 3, 0, 0, 0]], dtype=np.int32)
        c = np.array([[1, 4, 2, 5, 3, 6, 0, 0, 0, 0]], dtype=np.int32)

        # building graph
        q_input = Input((6,))
        c_input = Input((10,))
        q_len = SequenceLength()(q_input)
        c_len = SequenceLength()(c_input)
        embed = Embedding(7, 7, embeddings_initializer='identity')
        output = WordInQuestionW()([embed(q_input), embed(c_input), q_len, c_len])
        model = Model([q_input, c_input], output)

        # assertion
        result = model.predict([q, c])
        expected = np.array([0.61131245, 0.38868755, 0.61131245,
                             0.38868755, 0.61131245, 0.38868755, 0, 0, 0, 0],
                            dtype=np.float32)[None, :, None]
        np.testing.assert_almost_equal(result, expected)
