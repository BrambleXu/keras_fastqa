import numpy as np


def init_lstm_projection(shape, dtype=None):
    kernel_size, in_filters, out_filters = shape
    kernel = np.eye(in_filters, out_filters)
    kernel[out_filters:, :] = np.identity(out_filters)
    return kernel[None].astype(dtype)
