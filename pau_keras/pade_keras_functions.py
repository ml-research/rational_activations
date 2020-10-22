import tensorflow as tf


def get_xps(weight_denominator, weight_numerator, z):
    xps = list()
    xps.append(z)
    for _ in range(max(weight_numerator.shape[0], weight_denominator.shape[0])):
        xps.append(xps[-1] * z)
    xps.insert(0, tf.ones_like(z))
    return xps


def PAU_PYTORCH_A_F(z, weight_numerator, weight_denominator, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X ^ n /
    #               1 + | b_0 | | X | + | b_1 | | X | ^ 2 + ... + | b_i | | X | ^ {i + 1}

    xps = get_xps(weight_denominator, weight_numerator, z)

    numerator = 0
    for i in range(weight_numerator.shape[0]):
        w_n = weight_numerator[i]
        numerator = numerator + w_n * xps[i]

    denominator = 1.0
    for j in range(weight_denominator.shape[0]):
        w_d = weight_denominator[j]
        denominator = denominator + tf.abs(w_d * xps[j + 1])

    return numerator / denominator


def PAU_PYTORCH_B_F(z, weight_numerator, weight_denominator, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X ^ n /
    #               1 + |b_0*X + b_1*X^2 + ... + b_{n-1}*X^n|

    xps = get_xps(weight_denominator, weight_numerator, z)

    numerator = 0
    for i in range(weight_numerator.shape[0]):
        w_n = weight_numerator[i]
        numerator = numerator + w_n * xps[i]

    denominator = 0
    for j in range(weight_denominator.shape[0]):
        w_d = weight_denominator[j]
        denominator = denominator + w_d * xps[j + 1]

    return numerator / (1 + tf.abs(denominator))


def PAU_PYTORCH_C_F(z, weight_numerator, weight_denominator, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X ^ n /
    #               eps + |b_0*X + b_1*X^2 + ... + b_{n-1}*X^n|

    xps = get_xps(weight_denominator, weight_numerator, z)

    numerator = 0
    for i in range(weight_numerator.shape[0]):
        w_n = weight_numerator[i]
        numerator = numerator + w_n * xps[i]

    denominator = 0
    for j in range(weight_denominator.shape[0]):
        w_d = weight_denominator[j]
        denominator = denominator + w_d * xps[j + 1]

    return numerator / (0.1 + tf.abs(denominator))


def PAU_PYTORCH_D_F(x, weight_numerator, weight_denominator, training):
    raise NotImplementedError()
