import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pos = np.arange(seq_len)[:,np.newaxis]
    i = np.arange(d_model)[np.newaxis ,:]
    angle_rates = 1 / np.power(base, (2*(i//2)) / np.float64(d_model) )
    angle_rads = pos * angle_rates

    #Apply to sin to even and cos to odd indices
    pos_encoding = np.zeros(angle_rads.shape)
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return pos_encoding