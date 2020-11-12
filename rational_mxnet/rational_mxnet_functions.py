from mxnet import nd


def get_xps(weight_denominator, weight_numerator, z):
    xps = list()
    xps.append(z)
    for _ in range(max(len(weight_numerator), len(weight_denominator))):
        xps.append(nd.multiply(xps[-1], z))
    xps.insert(0, nd.ones_like(z))
    return xps


def Rational_MXNET_A_F(x, weight_numerator, weight_denominator, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X ^ n /
    #               1 + | b_0 * X | + | b_1 * X | ^ 2 + ... + | b_i * X | ^ {i + 1}

    z = nd.reshape(x, shape=(-1,))

    xps = get_xps(weight_denominator, weight_numerator, z)

    numerator = nd.array([0], dtype='float32')
    for i, w_n in enumerate(weight_numerator):
        numerator = numerator + nd.multiply(w_n, xps[i])

    denominator = nd.array([1.0], dtype='float32')
    for j, w_d in enumerate(weight_denominator):
        denominator = denominator + nd.abs(nd.multiply(w_d, xps[j + 1]))

    return nd.divide(numerator, denominator).reshape(x.shape)


def Rational_MXNET_B_F(x, weight_numerator, weight_denominator, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X ^ n /
    #               1 + |b_0*X + b_1*X^2 + ... + b_{n-1}*X^n|

    z = nd.reshape(x, shape=(-1,))

    xps = get_xps(weight_denominator, weight_numerator, z)

    numerator = nd.array([0], dtype='float32')
    for i, w_n in enumerate(weight_numerator):
        numerator = numerator + nd.multiply(w_n, xps[i])

    denominator = nd.array([0], dtype='float32')
    for j, w_d in enumerate(weight_denominator):
        denominator = denominator + nd.multiply(w_d, xps[j + 1])

    return nd.divide(numerator, (1.0 + nd.abs(denominator))).reshape(x.shape)


def Rational_MXNET_C_F(x, weight_numerator, weight_denominator, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X ^ n /
    #               eps + |b_1*X + b_1*X^2 + ... + b_{n-1}*X^n|

    z = nd.reshape(x, shape=(-1,))

    xps = get_xps(weight_denominator, weight_numerator, z)

    numerator = nd.array([0], dtype='float32')
    for i, w_n in enumerate(weight_numerator):
        numerator = numerator + nd.multiply(w_n, xps[i])

    denominator = nd.array([0], dtype='float32')
    for j, w_d in enumerate(weight_denominator):
        denominator = denominator + nd.multiply(w_d, xps[j])

    return nd.divide(numerator, (0.1 + nd.abs(denominator))).reshape(x.shape)


def Rational_MXNET_D_F(x, weight_numerator, weight_denominator, training, random_deviation=0.1):
    # P(X)/Q(X) = noised(a_0) + noised(a_1)*X +noised(a_2)*X^2 + ... + noised(a_n)*X^n /
    #                1 + |noised(b_0)*X + noised(b_1)*X^2 + ... + noised(b_{n-1})*X^n|
    # Noised parameters have uniform noise to be in range [(1-random_deviation)*parameter,(1+random_deviation)*parameter].

    if not training:
        # do not add noise
        return Rational_MXNET_B_F(x, weight_numerator, weight_denominator, training)

    z = nd.reshape(x, shape=(-1,))
    lower_bound = nd.array([1 - random_deviation])
    upper_bound = nd.array([1 + random_deviation])

    xps = get_xps(weight_denominator, weight_numerator, z)

    numerator = nd.array([0], dtype='float32')
    for i, w_n in enumerate(weight_numerator):
        w_n_noised = nd.multiply(w_n, nd.sample_uniform(low=lower_bound,
                                                        high=upper_bound,
                                                        shape=z.shape,
                                                        dtype='float32'))
        numerator = numerator + nd.multiply(w_n_noised, xps[i])

    denominator = nd.array([0], dtype='float32')
    for j, w_d in enumerate(weight_denominator):
        w_d_noised = nd.multiply(w_d, nd.sample_uniform(low=lower_bound,
                                                        high=upper_bound,
                                                        shape=z.shape,
                                                        dtype='float32'))
        denominator = denominator + nd.multiply(w_d_noised, xps[j + 1])

    return nd.divide(numerator, (1 + nd.abs(denominator))).reshape(x.shape)
