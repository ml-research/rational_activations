"""
This file contains the mathematical implementations of the rational activation function versions
a,b,c and d.
"""


def _get_xps_num(F, x, weights):
    """
    creates a sequence of ascending powers of x for numerator

    :param F: a function space either mxnet.nd or mxnet.sym
    :param x: input sequence of scalars
    :param weights: vector containing the weights a_0, ... a_n
    :return: a two-dimensional sequence that looks approximately like this
     [[1,1,...], [--x--], [--x^2--],... , [--x^n--]], where x is a vector (sequence of scalars)
    """
    #  create an array containing ones
    xps = F.expand_dims(F.ones_like(x), axis=0)

    # append arrays containing x, x^2, ... x^n to the list
    len_weights = int(F.shape_array(weights).asnumpy()[0])
    for i in range(len_weights - 1):
        factor = F.sum(F.ones(shape=(1, i + 1)))
        x_i = F.expand_dims(F.broadcast_power(x, factor), axis=0)
        xps = F.concat(xps, x_i, dim=0)

    return xps


def _get_xps_denom(F, x, weights):
    """
    creates a sequence of ascending powers of x for denominator

    :param F: a function space either mxnet.nd or mxnet.sym
    :param x: input sequence of scalars
    :param weights: vector containing the weights a_0, ... a_m
    :return: a two-dimensional sequence that looks approximately like this
     [[--x--], [--x^2--],... , [--x^n--], [--x^{m+1}--]], where x is a vector (sequence of scalars)
    """
    #  create an array containing x
    import mxnet as mx
    xps = F.expand_dims(F.elemwise_mul(x, F.ones_like(x)), axis=0)

    # append arrays containing x^2, ... x^{n+1} to the list
    len_weights = int(F.shape_array(weights).asnumpy()[0])
    for i in range(len_weights - 1):
        factor = F.sum(F.ones(shape=(1, i + 2)))
        x_i = F.expand_dims(F.broadcast_power(x, factor), axis=0)
        xps = F.concat(xps, x_i, dim=0)

    return xps


def _version_a(F, x, numerator_weights, denominator_weights, training):
    """
    version a of rational activation function

    f(x) = p(x) / q(x) = (a_0 + a_1 * x + a_2 * x^2 + ... + a_n * x^n) /
                (1 + |b_0 * x| + | b_1 * x^2| + ... + | b_m * x^{m+1}|)

    note: q(x) contains m absolute value terms here

    :param F: a function space either mxnet.nd or mxnet.sym
    :param x: input sequence of scalars
    :param numerator_weights: vector containing the weights a_0, ... a_n
    :param denominator_weights: vector containing the weights b_0, ... b_m
    :param training: (NOT IN USE) whether the call is in inference mode or training mode
    :return: f(x), i.e. the input tensor with the rational activation function applied to it
    """
    # get powers of x for numerator weights
    xps_num = _get_xps_num(F, x, numerator_weights)

    # multiply numerator weights with xps values, then sum them up
    numerator = F.sum(
        F.broadcast_mul(xps_num, F.expand_dims(numerator_weights, axis=1)), axis=0)

    # get powers of x for denominator weights
    xps_den = _get_xps_denom(F, x, denominator_weights)

    # in accordance with the formula (see docstring), a one-vector is added to the sum
    ones = F.ones_like(x)
    # multiply denominator weights with xps values calculate absolute value,
    # then sum them up and add the ones vector
    denominator = F.elemwise_add(ones, F.sum(F.abs(
        F.broadcast_mul(xps_den, F.expand_dims(denominator_weights, axis=1))), axis=0))

    return F.elemwise_div(numerator, denominator)


def _version_b(F, x, numerator_weights, denominator_weights, training):
    """
    version b of rational activation function

    f(x) = p(x) / q(x) = (a_0 + a_1 * x + a_2 * x^2 + ... + a_n * x^n) /
                (1 + |b_1 * x + b_2 * x^2 + ... + b_m * x^m|)

    note: q(x) contains only one absolute value term here

    :param F: a function space either mxnet.nd or mxnet.sym
    :param x: input sequence of scalars
    :param numerator_weights: vector containing the weights a_0, ... a_n
    :param denominator_weights: vector containing the weights b_0, ... b_m
    :param training: (NOT IN USE) whether the call is in inference mode or training mode
    :return: f(x), i.e. the input tensor with the rational activation function applied to it
    """
    """
    z = nd.reshape(x, shape=(-1,))

    xps = _get_xps(F, z, numerator_weights, denominator_weights)

    numerator = nd.array([0], dtype='float32')
    for i, w_n in enumerate(numerator_weights):
        numerator = numerator + nd.multiply(w_n, xps[i])

    denominator = nd.array([0], dtype='float32')
    for j, w_d in enumerate(denominator_weights):
        denominator = denominator + nd.multiply(w_d, xps[j + 1])

    return nd.divide(numerator, (1.0 + nd.abs(denominator))).reshape(x.shape)
    """
    return None


def _version_c(F, x, numerator_weights, denominator_weights, training):
    """
    version c of rational activation function

    f(x) = p(x) / q(x) = (a_0 + a_1 * x + a_2 * x^2 + ... + a_n * x^n) /
                (epsilon + |b_0 + b_1 * x + b_2 * x^2 + ... + b_m * x^m|)

    note: q(x) contains a variable term (epsilon) here, and also a b_0 term

    :param F: a function space either mxnet.nd or mxnet.sym
    :param x: input sequence of scalars
    :param numerator_weights: vector containing the weights a_0, ... a_n
    :param denominator_weights: vector containing the weights b_0, ... b_m
    :param training: (NOT IN USE) whether the call is in inference mode or training mode
    :return: f(x), i.e. the input tensor with the rational activation function applied to it
    """

    """
    z = nd.reshape(x, shape=(-1,))

    xps = _get_xps(F, z, numerator_weights, denominator_weights)

    # multiply numerator weights with xps values, then sum them up
    numerator = F.sum(F.broadcast_mul(numerator_weights, xps))

    # multiply denominator weights with xps values calculate absolute value,
    # then sum them up
    # NOTE THE INDEX CHANGE HERE, ACCOUNTING FOR THE '+1
    denominator = F.sum(F.broadcast_abs(
        F.broadcast_mul(denominator_weights, xps[1:])))

    return nd.divide(numerator, (0.1 + nd.abs(denominator))).reshape(x.shape)
    """
    return None


def _version_d(F, x, numerator_weights, denominator_weights, training, random_deviation=0.1):
    """
    version d of rational activation function

    f(x) = p(x) / q(x) =
    (noised(a_0) + noised(a_1) * x + noised(a_2) * x^2 + ... + noised(a_n) * x^n) /
                (1 + |noised(b_1) * x + noised(b_2) * x^2 + ... + noised(b_m) * X^m|)

    Noised parameters have uniform noise to be in range
    [(1-random_deviation)*parameter,(1+random_deviation)*parameter].

    :param F: a function space either mxnet.nd or mxnet.sym
    :param x: input sequence of scalars
    :param numerator_weights: vector containing the weights a_0, ... a_n
    :param denominator_weights: vector containing the weights b_0, ... b_m
    :param training: (NOT IN USE) whether the call is in inference mode or training mode
    :param random_deviation: random deviation
    :return: f(x), i.e. the input tensor with the rational activation function applied to it

    """

    """
    # if in training mode, apply Function B
    if not training:
        # do not add noise
        return _version_b(F, x, numerator_weights, denominator_weights, False)

    # else: inference mode
    # get list of polynomial [1, X, X^2, X^3....X^n]
    z = nd.reshape(x, shape=(-1,))
    lower_bound = nd.array([1 - random_deviation])
    upper_bound = nd.array([1 + random_deviation])

    xps = _get_xps(F, z, numerator_weights, denominator_weights)

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
    """
    return None
