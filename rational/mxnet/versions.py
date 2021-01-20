"""
TODO explain what this file is doing
"""
from mxnet import nd


def _get_xps(x, numerator_weights, denominator_weights):
    xps = list()
    xps.append(x)
    for _ in range(max(len(numerator_weights), len(denominator_weights))):
        xps.append(nd.multiply(xps[-1], x))
    xps.insert(0, nd.ones_like(x))
    return xps


def _version_a(x, numerator_weights, denominator_weights, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X ^ n /
    #               1 + | b_0 * X | + | b_1 * X | ^ 2 + ... + | b_i * X | ^ {i + 1}

    z = nd.reshape(x, shape=(-1,))

    xps = _get_xps(z, numerator_weights, denominator_weights)

    numerator = nd.array([0], dtype='float32')
    for i, w_n in enumerate(numerator_weights):
        numerator = numerator + nd.multiply(w_n, xps[i])

    denominator = nd.array([1.0], dtype='float32')
    for j, w_d in enumerate(denominator_weights):
        denominator = denominator + nd.abs(nd.multiply(w_d, xps[j + 1]))

    return nd.divide(numerator, denominator).reshape(x.shape)


def _version_b(x, numerator_weights, denominator_weights, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X ^ n /
    #               1 + |b_0*X + b_1*X^2 + ... + b_{n-1}*X^n|

    z = nd.reshape(x, shape=(-1,))

    xps = _get_xps(z, numerator_weights, denominator_weights)

    numerator = nd.array([0], dtype='float32')
    for i, w_n in enumerate(numerator_weights):
        numerator = numerator + nd.multiply(w_n, xps[i])

    denominator = nd.array([0], dtype='float32')
    for j, w_d in enumerate(denominator_weights):
        denominator = denominator + nd.multiply(w_d, xps[j + 1])

    return nd.divide(numerator, (1.0 + nd.abs(denominator))).reshape(x.shape)


def _version_c(x, numerator_weights, denominator_weights, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X ^ n /
    #               eps + |b_1*X + b_1*X^2 + ... + b_{n-1}*X^n|

    z = nd.reshape(x, shape=(-1,))

    xps = _get_xps(z, numerator_weights, denominator_weights)

    numerator = nd.array([0], dtype='float32')
    for i, w_n in enumerate(numerator_weights):
        numerator = numerator + nd.multiply(w_n, xps[i])

    denominator = nd.array([0], dtype='float32')
    for j, w_d in enumerate(denominator_weights):
        denominator = denominator + nd.multiply(w_d, xps[j])

    return nd.divide(numerator, (0.1 + nd.abs(denominator))).reshape(x.shape)


def _version_d(x, numerator_weights, denominator_weights, training, random_deviation=0.1):
    # P(X)/Q(X) = noised(a_0) + noised(a_1)*X +noised(a_2)*X^2 + ... + noised(a_n)*X^n /
    #                1 + |noised(b_0)*X + noised(b_1)*X^2 + ... + noised(b_{n-1})*X^n|
    # Noised parameters have uniform noise to be in range [(1-random_deviation)*parameter,(1+random_deviation)*parameter].

    if not training:
        # do not add noise
        return _version_b(x, numerator_weights, denominator_weights, training)

    z = nd.reshape(x, shape=(-1,))
    lower_bound = nd.array([1 - random_deviation])
    upper_bound = nd.array([1 + random_deviation])

    xps = _get_xps(z, numerator_weights, denominator_weights)

    numerator = nd.array([0], dtype='float32')
    for i, w_n in enumerate(numerator_weights):
        w_n_noised = nd.multiply(w_n, nd.sample_uniform(low=lower_bound,
                                                        high=upper_bound,
                                                        shape=z.shape,
                                                        dtype='float32'))
        numerator = numerator + nd.multiply(w_n_noised, xps[i])

    denominator = nd.array([0], dtype='float32')
    for j, w_d in enumerate(denominator_weights):
        w_d_noised = nd.multiply(w_d, nd.sample_uniform(low=lower_bound,
                                                        high=upper_bound,
                                                        shape=z.shape,
                                                        dtype='float32'))
        denominator = denominator + nd.multiply(w_d_noised, xps[j + 1])

    return nd.divide(numerator, (1 + nd.abs(denominator))).reshape(x.shape)
