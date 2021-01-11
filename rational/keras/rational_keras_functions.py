import tensorflow as tf


def get_xps(weight_denominator, weight_numerator, z):
    xps = list()
    xps.append(z)
    for _ in range(max(weight_numerator.shape[0], weight_denominator.shape[0])):
        xps.append(xps[-1] * z)
    xps.insert(0, tf.ones_like(z))
    return xps


def Rational_KERAS_A_F(z, weight_numerator, weight_denominator, training):
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


def Rational_KERAS_B_F(z, weight_numerator, weight_denominator, training):
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


def Rational_KERAS_C_F(z, weight_numerator, weight_denominator, training):
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
        denominator = denominator + w_d * xps[j]

    return numerator / (0.1 + tf.abs(denominator))


def Rational_KERAS_D_F(x, weight_numerator, weight_denominator, training, random_deviation=0.1):
    """
    version d of rational activation function
    return P(X) / Q(X) 
           the input tensor with Rational function applied to it

            P(X)/Q(X) = noised(a_0) + noised(a_1)*X +noised(a_2)*X^2 + ... + noised(a_n)*X^n /
            1 + |noised(b_0)*X + noised(b_1)*X^2 + ... + noised(b_{n-1})*X^n|
            Noised parameters have uniform noise to be in range [(1-random_deviation)*parameter,(1+random_deviation)*parameter]

    :param x: input tensor X
    :param weight_numerator: vector containing weights of numerator
    :param weight_denominator: vector containing weights of denominator
    :param training: whether the call is in inference mode or training mode
    :param random_deviation: deviation for determining range of noise parameters
    """

    # if in training mode, apply Function B
    if not training:
        return Rational_KERAS_B_F(x, weight_numerator, weight_denominator, training)

    # else: in inference mode
    else:
        # get list of polynomial [1, X, X^2, X^3....X^n]
        xps = get_xps(weight_denominator, weight_numerator, x)

        # assign weights to coefficients of numerator of polynomial
        numerator = 0
        for i in range(weight_numerator.shape[0]):
            # assign noise factor with uniform distribution
            noise = tf.random.uniform(
                shape=x.shape, minval=1-random_deviation, maxval=1+random_deviation, dtype=tf.dtypes.float32)
            w_n_noised = weight_numerator[i] * noise
            numerator = numerator + w_n_noised * xps[i]

        # assign weights to coefficients of denominator of polynomial
        denominator = 0
        for j in range(weight_denominator.shape[0]):
            noise = tf.random.uniform(
                shape=x.shape, minval=1-random_deviation, maxval=1+random_deviation, dtype=tf.dtypes.float32)
            w_d_noised = weight_denominator[j] * noise
            denominator = denominator + w_d_noised * xps[j]

        return numerator / (1 + tf.abs(denominator))
